# 32. Batching

By the end of this page many readers share each pass of the model, a request joins or leaves between two tokens, and each reader still gets the story they would get alone.

Lesson 31's machine served one reader while the rest stood in line, and each pass of the model wrote one token for that one reader.

## The idea

A pass of the model works on a grid of tokens with one row for each text. In training the grid had 32 rows. In writing it has had one. Give it a row for each reader, and one pass writes a token for each of them. The run below measures what that buys.

Three things stand in the way.

1. The rows have different lengths. One reader's text is 70 tokens and another's is 200, and a grid needs rows of one length. We line the rows up at their ends and put empty places in front of the shorter ones. A grid of true and false, called `real`, marks the places that hold tokens, and attention reads only those.
2. Readers come and go. A pass of the batch takes one new token for each row, and a prompt is many tokens. So a new request first reads its prompt alone, into a cache of its own, as in lesson 23. Then its cache joins the batch's cache as one more row. When a row writes its last token it is cut out of the grid.
3. Memory runs out. Each row's cache grows by lesson 23's 6,144 bytes a token. The engine takes a request only if the cache it may come to need, for its prompt and the most it may write, still fits in a budget. Otherwise it says no, and lesson 31's loop keeps that request in line until there is room.

Joining takes care with places. A row read its prompt at places 0, 1, 2 and so on, and lesson 15 turned each key by its place. In the batch that row may sit further along, behind some empty places. Rotary positions make the move cheap: a score depends only on the distance between two tokens. So we turn each of the row's keys on by the same number of places, and no score changes. A function called `slide` does that, and a test checks it.

A long prompt is read in chunks of 64 tokens, one chunk a round. The readers who are already being written to get their next token between two chunks.

## By hand

Row A has read 5 tokens and row B has read 3. B gets two empty places in front.

| Place | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Row A | a | a | a | a | a |
| Row B | empty | empty | b | b | b |
| `real` for B | false | false | true | true | true |

B's tokens were read at places 0, 1 and 2, and now sit at 2, 3 and 4. Each of B's keys is turned 2 places on. The next token of both rows goes to place 5. Alone, B's next query would sit at place 3 and meet its newest key at place 2, a distance of 1. In the batch the query is at 5 and that key at 4: a distance of 1 again.

Memory. A request with a prompt of 64 tokens that may write 64 more can come to hold 128 tokens.

128 × 6,144 = 786,432 bytes

Room for 16 such requests is 12.6 MB.

## The code

```python
# gpt/engine.py
def slide(kept, by, pad):
    """Put `pad` empty places before a cache and turn its keys `by` places on.

    Only the distance between two tokens enters a score (lesson 15), so a whole row can sit
    anywhere along the batch as long as its keys are turned to match.
    """
    for one in kept:
        k, v = one["k"], one["v"]
        places = torch.full((k.size(2),), by, device=k.device)
        empty = k.new_zeros(k.size(0), k.size(1), pad, k.size(3))
        one["k"], one["v"] = (
            torch.cat([empty, rope(k, places)], dim=2),
            torch.cat([empty, v], dim=2),
        )
```

`kept` is lesson 23's list of caches, one for each block. Values carry no place, so they are moved and not turned.

```python
# gpt/engine.py
    @torch.no_grad()
    def step(self):
        """One round of work. Returns each (request, token) written in it."""
        device, out = self.model.embed.weight.device, []
        if self.rows:  # one pass writes one token for every row
            last = torch.tensor([[r.written[-1]] for r in self.rows], device=device)
            self.real = torch.cat([self.real, self.real.new_ones(len(self.rows), 1)], dim=1)
            for one in self.kept:
                one["real"] = self.real
            scores = self.model(last, self.kept)[:, -1]
            for request, row in zip(self.rows, scores):
                request.written.append(choose(row, **request.how))
                out.append((request, request.written[-1]))
            self.leave()
        for request in list(self.reading):  # each new request reads one chunk of its prompt
            part = request.prompt[request.read : request.read + self.chunk]
            scores = self.model(torch.tensor([part], device=device), request.kept)[0, -1]
            request.read += len(part)
            if request.read == len(request.prompt):
                if self.store:
                    self.store.keep(request.prompt, request.kept)
                request.written.append(choose(scores, **request.how))
                out.append((request, request.written[-1]))
                self.reading.remove(request)
                if not request.done():
                    self.join(request)
        return out
```

`self.rows` are the requests being written to and `self.reading` the ones still reading their prompt. Each pass adds one real place to each row: the column of ones. Three short functions are not shown. `admit` does the sum of point 3, `join` lines a new row up with `slide`, and `leave` cuts out each row whose `done()` is true. Each row picks its own token with lesson 22's `choose`, with the settings in its own `request.how`, so two readers can ask for different temperatures. `self.store` is empty until lesson 33. The model itself gained two lines: when a cache holds `real`, attention's mask leaves out the places it marks false.

## The proof

```bash
uv run pytest tests/test_engine.py
```

- Five requests with prompts of different lengths share a batch, and each writes what it writes alone.
- The same five arrive in rounds 0, 2, 7 and 9, join a running batch and leave it at different times. Nothing any of them writes changes.
- A stop token ends its own row and no other.
- A prompt read in chunks of 4 tokens gives the answer it gives read whole, and a short request beside it gets its first token after one round.
- With four rows, one round calls the model once, on a grid of 4 rows and 1 new token each.
- With a budget for two requests, `admit` says no to a third, and yes once the first two are done.
- A row moved six places along, with its keys turned to match, gets the scores it got before.
- With the mask taken out, the empty places are read and the scores change.

```bash
uv run python -m gpt.engine
```

```
@@OUTPUT@@
```

Timings are from an M4 laptop that was running other jobs, and they move from run to run.

@@RESULTS@@

## What it still cannot do

Each request reads its whole prompt from the first token, even when a thousand requests before it began with the same words.

## Check yourself

Predict before you run: 16 readers share each pass. Does each one get their tokens faster or slower than a reader alone, and does the machine write more or fewer tokens a second in all?

<details><summary>Answer</summary>

Slower for each reader, more in all. A pass with 16 rows takes longer than a pass with one, and far less than 16 passes.
</details>

<details><summary>Row B joins a batch whose rows are 40 tokens long. B has read 25. How many empty places go in front of B, and how far are its keys turned?</summary>

15 empty places, and each key is turned 15 places on. B's tokens then sit at places 15 to 39, ending where the other rows end.
</details>

<details><summary>Why can a request not be added to the batch before it has read its prompt?</summary>

A pass of the batch reads one new token for each row. A prompt is many tokens, so it is read apart, into its own cache, and that cache joins as a row.
</details>

<details><summary>From lesson 15: what would break if positions came from lesson 8's table of places?</summary>

A row could not be moved. Lesson 8 added a place's numbers to the token's at the input, so each key made from them has place 3 mixed in, with no way to take it out. Each joining row would have to be read again at its new places.
</details>

Go deeper:

- Kwon and others, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023). The vLLM paper: the same batch, with the cache kept in blocks that rows can share.
- Yu and others, [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (2022). Where joining and leaving between two tokens comes from.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). A serving engine small enough to read in an evening.
