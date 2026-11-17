"""Lesson 36. The server: the model behind a web address, for many people at once."""

import codecs
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
        """Queue a prompt. Returns the request and the queue its tokens will come on, or None."""
        request = Request(self.words.encode(prompt), limit, stop=self.words.end, how=how)
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
        job = self.service.submit(
            ask["messages"][-1]["content"],
            ask.get("max_tokens", 250),
            temperature=ask.get("temperature", 0.8),
            top_p=ask.get("top_p", 1.0),
        )
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


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Storyteller</title>
<style>
  body { font: 18px/1.5 Georgia, serif; max-width: 34em; margin: 2em auto; padding: 0 1em; }
  input { font: inherit; width: 8em; margin: 0 .3em .3em 0; }
  button { font: inherit; padding: .2em .8em; }
  #story { white-space: pre-wrap; margin-top: 1em; min-height: 6em; }
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
<script>
const story = document.getElementById("story");
function say(text) {
  if (document.getElementById("aloud").checked && window.speechSynthesis)
    speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}
document.getElementById("go").onclick = async () => {
  const words = [...document.querySelectorAll(".word")].map(w => w.value).join(", ");
  const prompt = `Name: ${document.getElementById("name").value}\\nWords: ${words}\\nStory:`;
  story.textContent = "";
  if (window.speechSynthesis) speechSynthesis.cancel();
  const reply = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: [{role: "user", content: prompt}], stream: true}),
  });
  if (!reply.ok) { story.textContent = "The storyteller is busy. Try again in a moment."; return; }
  const reader = reply.body.getReader(), decoder = new TextDecoder();
  let rest = "", sentence = "";
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    rest += decoder.decode(value, {stream: true});
    const events = rest.split("\\n\\n");
    rest = events.pop();
    for (const event of events) {
      if (!event.startsWith("data: ") || event === "data: [DONE]") continue;
      const piece = JSON.parse(event.slice(6)).choices[0].delta.content;
      story.textContent += piece;
      sentence += piece;
      if (/[.!?]\\s*$/.test(sentence)) { say(sentence); sentence = ""; }
    }
  }
  if (sentence.trim()) say(sentence);
};
</script>
"""
