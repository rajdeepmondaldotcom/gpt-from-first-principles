# 36. The server

By the end of this page the storyteller answers over the web. You type a name and three words, a story streams back and is read aloud, you type what happens next, and other listeners are served at the same moment.

Each script so far runs once, in a terminal, and stops. Other programs cannot ask our model for a story.

## The idea

A server is a program that waits for requests from other programs, its clients, and answers each one. A web request is text sent to an address such as `http://127.0.0.1:8000/metrics`: a machine, a port on it, and a path. Ours serves the storyteller as lesson 29 left it, and has five jobs.

1. Speak a format that clients know. OpenAI's chat format is a list of messages sent to an address ending in `/v1/chat/completions`, answered in a text format called JSON. We answer it, so a tool written for it can use our model.
2. Stream. A listener wants each word as it is written, so the answer goes out as a run of small events, one for each piece of text.

   A token is not always a whole character: `é` is two bytes, and two tokens can carry one each. We hold the first half until the second arrives.
3. Serve many at once. Each web request gets a thread of its own: work that runs beside the others in one program and waits most of the time. One more thread runs lesson 32's engine. A request hands the engine its prompt through a queue, a line that one thread fills and another empties, and gets its tokens back through a second queue. A listener who leaves mid-story frees their row.
4. Say no early. A request the engine has no memory for yet waits in a line, as in lesson 31. The line has a limit. When it is full, the server answers "busy, try again" at once.
5. Count. A page at `/metrics` gives the running totals as plain text.

We check the counts two ways. An outside tool sends the load and measures it. One sum has to balance: requests inside the server, averaged over time, equal the rate at which they arrive times the average time one spends inside.

## By hand

Half a character. `é` is the two bytes 195 and 169.

| Token arrives | Bytes held | Sent to the listener |
|---|---|---|
| 195 | 195 | nothing |
| 169 | 195, 169 | `é` |

Requests inside. Say 5 requests arrive each second and each spends 2 seconds inside.

5 × 2 = 10 requests inside, on average

Over 100 seconds, 500 requests spend 2 seconds each, 1,000 request-seconds in all. Spread over 100 seconds, that is 10 at a time.

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
```

`self.inbox` is the waiting line, a queue with a limit. `held` is the request at its head while the engine has no memory for it. `self.outs` holds, for each request the engine has taken, the queue its tokens go to. Each time round, the thread takes in what there is room for, adds to the count of request-seconds, and runs one step of the engine.

If any of that fails, each answer being written ends with an error, and the thread starts a fresh engine.

The rest of the file is Python's standard web server, and `gpt/page.html` is the page you see in the browser. It reads each finished sentence aloud in the browser's own voice, and sends the story so far back with the line you type.

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
- A request with no prompt, a temperature that is text or `float('nan')`, or a limit of 0 tokens gets an error and the reason, and the next request is answered.
- A pass that fails gives each answer it was writing an error, begun or not, and the next request is answered.

The script gives the engine memory for 16 requests and a line of 64, asks for one story, then has `ab`, Apache's load tool, send 200 requests, 16 at a time.

Predict before you run: ab keeps 16 requests open at once. Averaged over the run, will the server count 16 requests inside?

```bash
uv run python -m gpt.server
```

```
one request over the web: 'Once upon a time there was a brave girl named Mira. She was always looking for something fun, so she wanted to buy a big'
200 requests, 16 at a time, by ab: 21.4 a second, 718 ms each
the same run by our own counts: 21.3 a second, 716 ms each
tokens written: 19,971, 2,131 a second
first token after 33 ms on average
requests inside, averaged over the run: 15.2
arrival rate times time inside: 15.3
a million tokens at that speed: 8 minutes of this machine
400 requests, 200 at a time: 100 answered, 300 told to try again, 0 failed
```

Timings are from an M4 laptop, and they move from run to run.

The outside tool and our own counts agree: 21.4 requests a second by its count and 21.3 by ours, and 718 ms each against 716. The server wrote 2,131 tokens a second, with a first token after 33 ms on average. Requests inside averaged 15.2. Requests left as fast as they arrived, at 21.3 a second, and 21.3 × 0.716 = 15.3: the sum holds.

At that speed a million tokens take this laptop 8 minutes. Sent 400 requests, 200 at a time, the server answered 100, told 300 to try again and failed none. That is more than memory for 16 and a line of 64 hold at once: as answers finished, later arrivals found room in the line.

To hear a story, start the server with `uv run python -m gpt.server stay` and open `http://127.0.0.1:8000`, which only this machine can reach.

## What it still cannot do

```
Once upon a time there was a brave girl named Mira. She was always looking for something fun, so she wanted to buy a big, beautiful fairy. One day she saw a beautiful, beautiful rainbow in the sky. She wanted to pick up the moon in her garden and take it home. 
Mira was so excited, she ran off to talk to the rainbow. Everywhere she went, she could see the beautiful rainbow. One day, the moon started to increase in size, and it got so big, it filled her garden.
Mira was so very brave, she didn’t even notice the moon! But Mira got trying to take the moon home, but she forgot about the moon. 
The moon was so far away, as the moon got too small and she could not get there. Mira was so scared she had to leave the moon in the night, and now the moon was back in the sky.
```

That story came over the web from our own server, to the order Mira, cat, moon and brave. It has Mira, the moon and brave, and no cat. Lesson 31 measured 539 tokens a second for one listener where memory alone allows 4,231, and lessons 34 and 35 made it no faster.

## Check yourself

<details><summary>Prediction</summary>

A little under: 15.2.
</details>

<details><summary>16 requests are inside at a time, each for 2 seconds. How many does the server finish each second?</summary>

8: the rate is 16 / 2.
</details>

<details><summary>From lesson 5: why can one token be half a character?</summary>

Our tokens are made of bytes, and a character outside plain English letters takes two bytes or more. Unless a merge joined them, each byte is a token of its own.
</details>

Go deeper:

- Google, [Site Reliability Engineering](https://sre.google/sre-book/handling-overload/), chapter 21. What a service should do when more arrives than it can serve.
- [vLLM](https://github.com/vllm-project/vllm). A server that speaks the same format, with the engine of lessons 32 and 33 built for graphics chips.
