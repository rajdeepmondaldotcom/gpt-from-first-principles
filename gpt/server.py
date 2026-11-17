"""Lesson 36. The server: the model behind a web address, for many people at once."""

import codecs
import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from gpt.engine import Engine, Request

FAILED = object()  # what a request's queue gets, in place of a token, when a pass has failed


def pieces(words, tokens):
    """Turn token ids into text as they arrive. Half a character waits for its other half."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token in tokens:
        text = decoder.decode(words.bytes[token]) if token != words.end else ""
        if text:
            yield text


class Service:
    """One thread runs the engine. Web requests hand it prompts and wait for tokens."""

    def __init__(self, model, words, budget, waiting=16, store=None):
        self.words, self.engine = words, Engine(model, budget, store=store)
        self.inbox = queue.Queue(maxsize=waiting)  # prompts the engine has not taken in yet
        self.outs = {}  # for each request in the engine: where its tokens go, and when it came
        self.counts = dict.fromkeys(
            [
                "requests",
                "refused",
                "failed",
                "finished",
                "tokens",
                "first_token_seconds",
                "seconds_in_system",
                "in_flight_seconds",
            ],
            0,
        )
        self.lock = threading.Lock()  # web threads and the engine's thread both count refusals
        self.running = True
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def submit(self, prompt, limit, temperature=0.8, top_p=1.0):
        """Queue a prompt. Returns the request and the queue its tokens will come on, or None
        when the line is full. A ValueError says why a request cannot be served at all.

        The model never trained past its context, so the answer stops there.
        """
        numbers = (limit, temperature, top_p)
        if not isinstance(prompt, str) or any(type(x) not in (int, float) for x in numbers):
            raise ValueError("the prompt must be text, and the settings numbers")
        # Written so that a setting that is no number at all (JSON allows NaN) is turned down too.
        if not (limit >= 1 and (temperature == 0 or temperature >= 0.01) and 0 < top_p <= 1):
            raise ValueError(
                "max_tokens starts at 1, temperature is 0 or at least 0.01, and top_p is above 0 to 1"
            )
        ids = self.words.encode(prompt)
        room = self.engine.model.context - len(ids)
        if not ids or room < 1:
            raise ValueError("the prompt is empty, or fills the model's context")
        how = {"temperature": temperature, "top_p": top_p}
        request = Request(ids, int(min(limit, room)), stop=self.words.end, how=how)
        out = queue.Queue()
        try:
            self.inbox.put_nowait((request, out, time.perf_counter()))
        except queue.Full:  # too many waiting already: say no now, not after a long wait
            with self.lock:
                self.counts["refused"] += 1
            return None
        return request, out

    def work(self):
        last, held = time.perf_counter(), None  # held: the head of the line, while it has no room
        while self.running:
            try:
                while held or not self.inbox.empty():
                    if not held:
                        held = self.inbox.get()
                        self.counts["requests"] += 1
                    request, out, arrived = held
                    if self.engine.admit(request):
                        self.outs[id(request)] = (out, arrived)
                    elif self.outs:
                        break  # no memory yet: it waits, and so does everyone behind it
                    else:  # it would not fit in an empty engine
                        with self.lock:
                            self.counts["refused"] += 1
                        out.put(None)
                    held = None
                now = time.perf_counter()
                inside = len(self.outs) + self.inbox.qsize() + bool(held)  # written to, or in line
                self.counts["in_flight_seconds"] += inside * (now - last)
                last = now
                if not self.outs:
                    time.sleep(0.001)
                    continue
                written = self.engine.step()
            except Exception:  # noqa: BLE001  one failure must not end the service
                traceback.print_exc()
                lost = [out for out, _ in self.outs.values()] + ([held[1]] if held else [])
                self.counts["failed"] += len(lost)
                for out in lost:
                    out.put(FAILED)  # in place of a token: this answer is over, and not whole
                self.outs.clear()
                held, engine = None, self.engine
                self.engine = Engine(engine.model, engine.budget, store=engine.store)
                continue
            for request, token in written:
                out, arrived = self.outs[id(request)]
                now = time.perf_counter()
                if len(request.written) == 1:
                    self.counts["first_token_seconds"] += now - arrived
                self.counts["tokens"] += 1
                out.put(token)
                if request.done():  # counted before the stream ends, so the counts are never behind
                    self.counts["finished"] += 1
                    self.counts["seconds_in_system"] += now - arrived
                    del self.outs[id(request)]
                    out.put(None)  # None ends the stream
        for out in [held and held[1], *(out for out, _ in self.outs.values())]:
            if out:
                out.put(None)  # stopped: nobody is left waiting on a stream that will not come

    def stop(self):
        """Stop the engine's thread, and end each stream that was waiting on it."""
        self.running = False
        self.thread.join()
        while not self.inbox.empty():
            self.inbox.get()[1].put(None)

    def metrics(self):
        """The counts as plain text, one to a line, in the form monitoring tools read."""
        lines = [f"stories_{name} {value}" for name, value in self.counts.items()]
        return "\n".join([*lines, f"stories_being_written {len(self.outs)}", ""])


class Handler(BaseHTTPRequestHandler):
    service = None

    def log_message(self, *args):  # no line on the terminal for each request
        pass

    def reply(self, status, kind, body):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/metrics":
            self.reply(200, "text/plain", self.service.metrics().encode())
        elif self.path == "/":
            page = Path(__file__).with_name("page.html").read_bytes()  # the storyteller's page
            self.reply(200, "text/html; charset=utf-8", page)
        else:
            self.reply(404, "text/plain", b"not found\n")

    def do_POST(self):
        """The chat API most tools speak: messages in, text out, as one answer or as a stream."""
        if self.path != "/v1/chat/completions":
            return self.reply(404, "text/plain", b"not found\n")
        try:
            ask = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            job = self.service.submit(
                ask["messages"][-1]["content"],
                ask.get("max_tokens", 250),
                ask.get("temperature", 0.8),
                ask.get("top_p", 1.0),
            )
        except (ValueError, KeyError, IndexError, TypeError) as problem:
            # Nothing we can do with this request: say why, and keep serving the others.
            body = json.dumps({"error": f"{type(problem).__name__}: {problem}"}).encode()
            return self.reply(400, "application/json", body + b"\n")
        first = job[1].get() if job else None
        if first is None:
            return self.reply(503, "application/json", b'{"error": "busy, try again"}\n')
        request, out = job
        failed = b'{"error": "the model failed on this request"}\n'
        text = pieces(self.service.words, self.live(first, out))
        if not ask.get("stream"):
            try:
                answer = {"role": "assistant", "content": "".join(text)}
            except RuntimeError:  # a pass failed before the answer was whole
                return self.reply(500, "application/json", failed)
            body = {"object": "chat.completion", "choices": [{"index": 0, "message": answer}]}
            return self.reply(200, "application/json", json.dumps(body).encode())
        if first is FAILED:
            return self.reply(500, "application/json", failed)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for piece in text:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": piece}}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except RuntimeError:  # a pass failed part way: say so where [DONE] would have come
            self.wfile.write(b"data: " + failed + b"\n")
        except (BrokenPipeError, ConnectionResetError):  # the reader left: stop writing for them
            request.limit = 0
        return None

    @staticmethod
    def live(first, out):
        """One answer's tokens as they come. The engine sends None after the last one, and
        FAILED in place of a token when a pass failed."""
        token = first
        while token is not None:
            if token is FAILED:
                raise RuntimeError("a pass of the model failed")
            yield token
            token = out.get()


class Server(ThreadingHTTPServer):
    request_queue_size = 256  # a burst of new connections waits here: the default holds 5


def serve(service, port=8000):
    """Start answering on this machine only. Returns the server, so a caller can shut it down."""
    Handler.service = service
    server = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    import urllib.request
    from pathlib import Path

    from gpt.cost import cache_bytes
    from gpt.prefix import PrefixStore
    from gpt.results import save, say
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "preferred")  # the storyteller after lesson 29's pairs
    room = cache_bytes(model, model.context)  # the cache of one request that fills the context
    # Memory for 16 requests that fill the context, a line of 64, and a store of beginnings as
    # large as 8 such caches: a listener who asks what happens next sends the story so far again.
    service = Service(model, words, 16 * room, waiting=64, store=PrefixStore(8 * room))
    server = serve(service)
    url = "http://127.0.0.1:8000/v1/chat/completions"
    if sys.argv[1:] == ["stay"]:
        print("the storyteller is at http://127.0.0.1:8000 until you press Ctrl-C")
        threading.Event().wait()

    prompt = "Name: Mira\nWords: cat, moon, brave\nStory:"
    ask = {"messages": [{"role": "user", "content": prompt}]}
    whole = urllib.request.Request(url, json.dumps(ask).encode())  # no limit: a story to its end
    with urllib.request.urlopen(whole, timeout=60) as reply:
        story = json.loads(reply.read())["choices"][0]["message"]["content"]
    body = json.dumps(ask | {"max_tokens": 100})  # the load test asks for 100 tokens each time
    say(f"one request over the web: {story.strip()[:120]!r}")
    assert shutil.which("ab"), "this script measures with ab, the load tool that ships with Apache"

    def load_test(count, at_once):
        """Send `count` requests, `at_once` at a time, with ab. Returns its report and our counts."""
        before, began = dict(service.counts), time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
            file.write(body)
            file.flush()
            ask = ["ab", "-q", "-n", str(count), "-c", str(at_once), "-s", "120", "-p", file.name]
            report = subprocess.run(
                [*ask, "-T", "application/json", url], capture_output=True, text=True, check=True
            ).stdout
        took = time.perf_counter() - began
        return report, {name: service.counts[name] - before[name] for name in before}, took

    report, ours, took = load_test(200, 16)
    theirs_rate = float(re.search(r"Requests per second:\s+([\d.]+)", report)[1])
    # ab's table of times has a row for the whole request: the least it took, then the mean.
    theirs_each = float(re.search(r"Total:\s+\d+\s+(\d+)", report)[1])
    rate, each = ours["finished"] / took, ours["seconds_in_system"] / ours["finished"]
    say(
        f"200 requests, 16 at a time, by ab: {theirs_rate:.1f} a second, {theirs_each:,.0f} ms each"
    )
    say(f"the same run by our own counts: {rate:.1f} a second, {each * 1e3:,.0f} ms each")
    say(f"tokens written: {ours['tokens']:,}, {ours['tokens'] / took:,.0f} a second")
    say(
        f"first token after {ours['first_token_seconds'] / ours['finished'] * 1e3:,.0f} ms on average"
    )
    say(f"requests inside, averaged over the run: {ours['in_flight_seconds'] / took:.1f}")
    say(f"arrival rate times time inside: {rate * each:.1f}")
    minutes = 1e6 / (ours["tokens"] / took) / 60
    say(f"a million tokens at that speed: {minutes:.0f} minutes of this machine")

    _, burst, _ = load_test(400, 200)  # far more at once than the line of 64 holds
    say(
        f"400 requests, 200 at a time: {burst['finished']} answered, "
        f"{burst['refused']} told to try again, {burst['failed']} failed"
    )
    save(
        "36_server",
        torch=torch.__version__,
        device="cpu",
        ab={"requests_a_second": theirs_rate, "ms_each": theirs_each},
        ours={"requests_a_second": rate, "ms_each": each * 1e3, **ours},
        burst=burst,
        sample=story.strip(),
    )
