# 36. The server

By the end of this page the storyteller answers over the web. You type a name and three words, a story streams back and is read aloud, you say what happens next, and other listeners are served at the same moment.

Each script so far runs once, in a terminal, and stops. Nobody else can ask our model for a story.

## The idea

A server is a program that waits for requests from other programs, its clients, and answers each one. Ours has five jobs.

1. Speak a format that clients know. Most tools that talk to a language model send the same request: a list of messages, sent to a web address that ends in `/v1/chat/completions`, answered in a text format called JSON. We answer that request, so any such tool can use our model, and so can a tool built to test servers under load.
2. Stream. A story takes seconds to write, and a listener wants each word as it comes. So the answer goes out as a run of small events, one for each piece of text. A token is not always a whole character: `é` is two bytes, and two tokens can carry one byte each. We hold the first half back until the second arrives.
3. Serve many at once. A thread is a line of work that runs beside others inside one program. Each web request gets a thread of its own, which mostly waits. One more thread runs lesson 32's engine. A queue is a line that one thread puts things into and another takes them out of. A request hands the engine its prompt through a queue and gets its tokens back through another.
4. Say no early. A request the engine has no memory for yet waits in a line, as in lesson 31. The line has a limit. When it is full, the server answers "busy, try again" at once. A fast no is kinder than a slow one. When a listener leaves in the middle of a story, the server stops writing for them and frees their row.
5. Count. A page at `/metrics` gives the running totals as plain text: requests, refusals, tokens, the wait for first tokens and the time requests spent inside.

Two checks tell us the counts are right. An outside tool sends the load and measures it from its own side. And one sum has to balance: the number of requests inside the server, averaged over time, equals the rate at which they arrive times the average time each one spends inside.

## By hand

Half a character. `é` is the two bytes 195 and 169.

| Token arrives | Bytes held | Sent to the listener |
|---|---|---|
| 195 | 195 | nothing |
| 169 | 195, 169 | `é` |

Requests inside. Say 5 requests arrive each second and each spends 2 seconds inside.

5 × 2 = 10 requests inside, on average

A way to see it: over 100 seconds, 500 requests spend 2 seconds each, 1,000 request-seconds in all. Spread over 100 seconds, that is 10 at a time.

## The code

```python
# gpt/server.py
def pieces(words, tokens):
    """Turn token ids into text as they arrive. Half a character waits for its other half."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token in tokens:
        text = decoder.decode(words.bytes[token]) if token != words.end else ""
        if text:
            yield text
```

`words.bytes[token]` is lesson 5's table from a token to its bytes. `codecs.getincrementaldecoder("utf-8")` is Python's own tool for bytes that arrive a few at a time. It gives back text only for the characters it has all of, and `errors="replace"` puts a mark where bytes make no character.

The engine's thread:

```python
# gpt/server.py
    def work(self):
        last, held = time.perf_counter(), None  # held: the head of the line, while it has no room
        while self.running:
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
                    self.counts["refused"] += 1
                    out.put(None)
                held = None
            now = time.perf_counter()
            inside = (
                len(self.outs) + self.inbox.qsize() + bool(held)
            )  # being written to, or in line
            self.counts["in_flight_seconds"] += inside * (now - last)
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
```

`self.inbox` is the waiting line, a queue that holds a set number of requests and no more. `held` is the request at its head while the engine has no memory for it. `self.outs` holds, for each request the engine has taken, the queue its tokens go to. Each time round, the thread takes in what there is room for, adds to the count of request-seconds, and runs one step of the engine. The rest of the file is Python's standard web server, the no for a full line, the stop for a listener who has left, and the page you see in the browser.

## The proof

```bash
uv run pytest tests/test_server.py
```

- Half of `é` sends nothing, and both halves send `é`.
- An answer over the web is the answer the model gives alone, whole or streamed.
- Six people who ask at the same moment each get their own answer.
- An answer stops at the model's context, and a prompt that fills the context is refused with the reason.
- A listener who leaves after one piece frees their row.
- A request too big for the whole memory gets a no at once. With memory for one request, a second waits and is then answered. With a line of two full, the third is turned away.
- The metrics page counts what happened, and requests inside added up over time equal time inside added up over requests.

The script starts the server, asks it for one story, then has `ab`, a load tool that comes with the Apache web server, send 200 requests, 16 at a time.

```bash
uv run python -m gpt.server
```

```
@@OUTPUT@@
```

Timings are from an M4 laptop that was running other jobs, and they move from run to run.

@@RESULTS@@

To keep it running and hear a story, start it with `uv run python -m gpt.server stay` and open `http://127.0.0.1:8000`. That address is this machine itself, at door 8000, and nothing outside the machine can reach it.

## What it still cannot do

```
@@SAMPLE@@
```

That story came over the web from our own server. The server gives the right answers, and we wrote it to be read, in Python. We have not asked how fast an engine built for speed would run the same model.

## Check yourself

Predict before you run: 16 requests are inside the server at a time, each takes 2 seconds, and they leave as fast as they arrive. How many does the server finish each second?

<details><summary>Answer</summary>

8. Requests inside equal the rate times the time inside, so the rate is 16 / 2.
</details>

<details><summary>Why does the server refuse a request when the line is full, when it could let the line grow?</summary>

Lesson 31 showed what a growing line does: each new request waits longer than the last. A no that comes at once lets the client try again or go elsewhere.
</details>

<details><summary>A listener closes the page halfway through a story. What would it cost to go on writing?</summary>

A row of the batch and its cache, until the story ends, for text nobody reads. Each other listener's tokens would come a little slower.
</details>

<details><summary>From lesson 5: why can one token be half a character?</summary>

Our tokens are made of bytes, and a character outside plain English letters takes two bytes or more. Until a merge joins them, each byte is a token of its own.
</details>

Go deeper:

- OpenAI, [the API written down as a specification](https://github.com/openai/openai-openapi). The chat completions request and the stream of events that our server answers with.
- Google, [Site Reliability Engineering](https://sre.google/sre-book/handling-overload/), chapter 21. What a service should do when more arrives than it can serve.
- [vLLM](https://github.com/vllm-project/vllm). A server that speaks the same format, with the engine of lessons 32 and 33 built for graphics chips.
