"""Lesson 36. The server: the model behind a web address, for many people at once."""

import codecs
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from gpt.engine import Engine, Request


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
                "finished",
                "tokens",
                "first_token_seconds",
                "seconds_in_system",
                "in_flight_seconds",
            ],
            0,
        )
        self.running = True
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, prompt, limit, **how):
        """Queue a prompt. Returns the request and the queue its tokens will come on, or None.

        The model never trained past its context, so the answer stops there. A prompt that fills
        the context leaves no room to write: a ValueError.
        """
        ids = self.words.encode(prompt)
        room = self.engine.model.context - len(ids)
        if room < 1:
            raise ValueError("the prompt fills the model's context")
        request = Request(ids, min(limit, room), stop=self.words.end, how=how)
        out = queue.Queue()
        try:
            self.inbox.put_nowait((request, out, time.perf_counter()))
        except queue.Full:  # too many waiting already: say no now, not after a long wait
            self.counts["refused"] += 1
            return None
        return request, out

    def work(self):
        last = time.perf_counter()
        while self.running:
            while not self.inbox.empty():
                request, out, arrived = self.inbox.get()
                self.counts["requests"] += 1
                if self.engine.admit(request):
                    self.outs[id(request)] = (out, arrived)
                else:  # no memory for it
                    self.counts["refused"] += 1
                    out.put(None)
            now = time.perf_counter()
            self.counts["in_flight_seconds"] += len(self.outs) * (now - last)
            last = now
            if not self.outs:
                time.sleep(0.001)
                continue
            for request, token in self.engine.step():
                out, arrived = self.outs[id(request)]
                now = time.perf_counter()
                if len(request.written) == 1:
                    self.counts["first_token_seconds"] += now - arrived
                self.counts["tokens"] += 1
                out.put(token)
                if request.done():
                    out.put(None)  # None ends the stream
                    self.counts["finished"] += 1
                    self.counts["seconds_in_system"] += now - arrived
                    del self.outs[id(request)]

    def metrics(self):
        """The counts as plain text, one to a line, in the form monitoring tools read."""
        lines = [f"stories_{name} {value}" for name, value in self.counts.items()]
        return "\n".join([*lines, f"stories_in_flight {len(self.outs)}", ""])


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
            self.reply(200, "text/html; charset=utf-8", PAGE.encode())
        else:
            self.reply(404, "text/plain", b"not found\n")

    def do_POST(self):
        """The chat API most tools speak: messages in, text out, as one answer or as a stream."""
        if self.path != "/v1/chat/completions":
            return self.reply(404, "text/plain", b"not found\n")
        ask = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            job = self.service.submit(
                ask["messages"][-1]["content"],
                ask.get("max_tokens", 250),
                temperature=ask.get("temperature", 0.8),
                top_p=ask.get("top_p", 1.0),
            )
        except ValueError as problem:  # nothing we can do with this request: say why
            body = json.dumps({"error": str(problem)}).encode()
            return self.reply(400, "application/json", body + b"\n")
        first = job[1].get() if job else None
        if first is None:
            return self.reply(503, "application/json", b'{"error": "busy, try again"}\n')
        request, out = job
        text = pieces(
            self.service.words,
            [first, *iter(out.get, None)] if not ask.get("stream") else self.live(first, out),
        )
        if not ask.get("stream"):
            answer = {"role": "assistant", "content": "".join(text)}
            body = {"object": "chat.completion", "choices": [{"index": 0, "message": answer}]}
            return self.reply(200, "application/json", json.dumps(body).encode())
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
        except (BrokenPipeError, ConnectionResetError):  # the reader left: stop writing for them
            request.limit = 0
        return None

    @staticmethod
    def live(first, out):
        yield first
        yield from iter(out.get, None)


def serve(service, port=8000):
    """Start answering on this machine only. Returns the server, so a caller can shut it down."""
    Handler.service = service
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Storyteller</title>
<style>
  body { font: 18px/1.5 Georgia, serif; max-width: 34em; margin: 2em auto; padding: 0 1em; }
  input { font: inherit; width: 8em; margin: 0 .3em .3em 0; }
  button { font: inherit; padding: .2em .8em; }
  #story { white-space: pre-wrap; margin-top: 1em; min-height: 6em; }
  #next { width: 20em; }
</style>
<h1>Tell me a story</h1>
<p>
  <input id="name" placeholder="a name" value="Mira">
  <input class="word" placeholder="a word" value="cat">
  <input class="word" placeholder="a word" value="moon">
  <input class="word" placeholder="a word" value="brave">
  <button id="go">Tell it</button>
  <label><input type="checkbox" id="aloud" style="width:auto" checked> read it aloud</label>
</p>
<div id="story"></div>
<p id="more" hidden>
  <input id="next" placeholder="What happens next?">
  <button id="on">Go on</button>
</p>
<script>
const story = document.getElementById("story"), more = document.getElementById("more");
let text = "";  // everything the model has read and written so far
function say(words) {
  if (document.getElementById("aloud").checked && window.speechSynthesis)
    speechSynthesis.speak(new SpeechSynthesisUtterance(words));
}
async function tell(limit) {
  more.hidden = true;
  const reply = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: [{role: "user", content: text}], stream: true, max_tokens: limit}),
  });
  if (reply.status === 400) { story.textContent += "\n\nThe end: this story is as long as the model can read."; return; }
  if (!reply.ok) { story.textContent += "\n\nThe storyteller is busy. Try again in a moment."; return; }
  const reader = reply.body.getReader(), decoder = new TextDecoder();
  let rest = "", sentence = "";
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    rest += decoder.decode(value, {stream: true});
    const events = rest.split("\n\n");
    rest = events.pop();
    for (const event of events) {
      if (!event.startsWith("data: ") || event === "data: [DONE]") continue;
      const piece = JSON.parse(event.slice(6)).choices[0].delta.content;
      story.textContent += piece;
      text += piece;
      sentence += piece;
      if (/[.!?]\s*$/.test(sentence)) { say(sentence); sentence = ""; }
    }
  }
  if (sentence.trim()) say(sentence);
  more.hidden = false;
}
document.getElementById("go").onclick = () => {
  const words = [...document.querySelectorAll(".word")].map(w => w.value).join(", ");
  text = `Name: ${document.getElementById("name").value}\nWords: ${words}\nStory:`;
  story.textContent = "";
  if (window.speechSynthesis) speechSynthesis.cancel();
  tell(120);
};
document.getElementById("on").onclick = () => {
  const next = " " + document.getElementById("next").value.trim();
  story.textContent += next;
  text += next;
  say(next);
  document.getElementById("next").value = "";
  tell(100);
};
</script>
"""

if __name__ == "__main__":
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    import urllib.request
    from pathlib import Path

    from gpt.cost import cache_bytes
    from gpt.results import save, say
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "storyteller")  # lesson 25's model, which writes to order
    service = Service(model, words, budget=16 * cache_bytes(model, model.context), waiting=64)
    server = serve(service)
    url = "http://127.0.0.1:8000/v1/chat/completions"
    if sys.argv[1:] == ["stay"]:
        print("the storyteller is at http://127.0.0.1:8000 until you press Ctrl-C")
        threading.Event().wait()

    prompt = "Name: Mira\nWords: cat, moon, brave\nStory:"
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 100})
    with urllib.request.urlopen(urllib.request.Request(url, body.encode()), timeout=60) as reply:
        story = json.loads(reply.read())["choices"][0]["message"]["content"]
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
    say(f"requests in flight, averaged over the run: {ours['in_flight_seconds'] / took:.1f}")
    say(f"arrival rate times time in system: {rate * each:.1f}")
    minutes = 1e6 / (ours["tokens"] / took) / 60
    say(f"a million tokens at that speed: {minutes:.0f} minutes of this machine")

    report, burst, _ = load_test(400, 200)  # far more at once than the line of 64 holds
    turned_away = int(re.search(r"Non-2xx responses:\s+(\d+)", report)[1])
    say(
        f"400 requests, 200 at a time: {burst['finished']} answered, {turned_away} told to try again"
    )
    save(
        "36_server",
        torch=torch.__version__,
        device="cpu",
        ab={"requests_a_second": theirs_rate, "ms_each": theirs_each},
        ours={"requests_a_second": rate, "ms_each": each * 1e3, **ours},
        burst={"answered": burst["finished"], "turned_away": turned_away},
        sample=story.strip(),
    )
