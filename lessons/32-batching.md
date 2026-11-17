# 32. Batching

By the end of this page many listeners share each pass of the model, a request joins or leaves between two tokens, and each listener still gets the story they would get alone.

Lesson 31's machine served one listener while the rest stood in line, and each pass of the model wrote one token for that one listener.

## The idea

A pass of the model works on a grid of tokens with one row for each text. In training the grid had 32 rows. In writing it has had one. Give it a row for each listener, and one pass writes a token for each of them.

1. The rows have different lengths, and a grid needs rows of one length. We line the rows up at their ends and put empty places in front of the shorter ones. A grid of true and false, called `real`, marks the places that hold tokens, and attention reads only those.
2. Listeners come and go. A pass of the batch takes one new token for each row, and a prompt is many tokens. So a new request first reads its prompt alone, into a cache of its own, as in lesson 23. Then its cache joins the batch's cache as one more row. When a row writes its last token it is cut out of the grid.
3. Memory runs out. Each row's cache grows by lesson 23's 6,144 bytes a token, and the grid is as long as its longest row. So the engine counts each request at the size of the longest, and takes a new one only if the rows together still fit in a budget. Otherwise it says no, and lesson 31's loop keeps that request in line until there is room.

A row read its prompt at places 0, 1, 2 and so on, and the model turned each key by its place, as in lesson 15. In the batch that row may sit further along, behind some empty places. With rotary positions a score depends only on the distance between two tokens, so we turn each of the row's keys further by the same number of places and no score changes.

The engine works in rounds. A round writes one token for each row and reads 64 tokens of each prompt still being read.

## By hand

Row A has read 5 tokens and row B has read 3. B gets two empty places in front.

| Place | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Row A | a | a | a | a | a |
| Row B | empty | empty | b | b | b |
| `real` for B | false | false | true | true | true |

B's tokens were read at places 0, 1 and 2, and now sit at 2, 3 and 4, so each of B's keys is turned 2 more places. Alone, B's next query would sit at place 3 and meet its newest key at place 2, a distance of 1. In the batch the query is at 5 and that key at 4: a distance of 1 again.

Memory. A request with a prompt of 64 tokens that may write 64 more can hold 128 tokens.

128 × 6,144 = 786,432 bytes

Room for 16 such requests is 12.6 MB. A request of 16 tokens beside them counts as 128 too.

## The code

```python
# gpt/engine.py
def slide(kept, by, pad):
    """Put `pad` empty places before a cache and turn its keys `by` places further.

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

`kept` is lesson 23's list of caches, one for each block. Values carry no place, so we move them without turning them.

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

`self.rows` are the requests being written to and `self.reading` the ones still reading their prompt. Each pass adds one real place to each row: the column of ones.

`admit` does the sum of point 3, `join` lines a new row up with `slide`, and `leave` cuts out each row whose `done()` is true. Each row picks its token with lesson 22's `choose` and its own settings, `request.how`. `self.store` is empty until lesson 33.

The model itself gained two lines: when a cache holds `real`, attention's mask leaves out the places it marks false.

## The proof

```bash
uv run pytest tests/test_engine.py
```

- Five requests with prompts of different lengths arrive in rounds 0, 2, 7 and 9, join a running batch and leave it at different times. Each gets, at each token, the scores the model gives that text read alone.
- A stop token ends its own row and no other.
- A prompt read in chunks of 4 tokens gives the answer it gives read whole, and a short request beside it gets its first token after one round.
- With four rows, one round calls the model once, on a grid of 4 rows and 1 new token each.
- With a budget for two requests, `admit` says no to a third, and yes once the first two are done. Beside a prompt of 40 tokens, a prompt of 2 counts as long, and the cache never passes the budget.
- A row moved six places along, with its keys turned to match, gets the scores it got before.
- With the mask taken out, the empty places are read and the scores change.

```bash
uv run python -m gpt.engine
```

```
8 requests that join a running batch two steps apart: 8 write what they write alone
a prompt of 192 tokens read in chunks of 64: the same answer, True
1 at once: 657 tokens a second in all, 669 for each listener
2 at once: 805 tokens a second in all, 414 for each listener
4 at once: 1,180 tokens a second in all, 308 for each listener
8 at once: 1,617 tokens a second in all, 217 for each listener
16 at once: 2,020 tokens a second in all, 139 for each listener
32 at once: 2,275 tokens a second in all, 80 for each listener
lesson 31's machine finished 10.1 requests a second, one at a time
the cache for one request of 64 + 64 tokens: 786,432 bytes
5.0 requests a second, room for 16: first token after 3 ms for half, 7 ms for 95 in 100, 276 tokens a second in all
9.1 requests a second, room for 16: first token after 6 ms for half, 9 ms for 95 in 100, 493 tokens a second in all
15.1 requests a second, room for 16: first token after 8 ms for half, 12 ms for 95 in 100, 813 tokens a second in all
a request offered one byte less than its cache needs is taken: False
```

Timings are from an M4 laptop, and they move from run to run.

The eight requests each write what they write alone, and so does the prompt read in chunks.

Alone, a listener gets 669 tokens a second, or 657 with the reading of the prompt counted. Among 16, each gets 139 and the machine writes 2,020, 3.1 times what it wrote for one. At 32 it writes 2,275, 13% more than at 16.

At 15.1 requests a second, half of lesson 31's listeners waited 1,737 ms for a first token. With room for 16, half wait 8 ms and 95 in 100 wait 12 ms or less, and the machine writes 813 tokens a second where it wrote 631.

## What it still cannot do

Each request reads its whole prompt from the first token, even when the requests before it began with the same words.

## Check yourself

Predict before you run: 16 listeners share each pass. Does each one get their tokens faster or slower than a listener alone, and does the machine write more or fewer tokens a second in all?

<details><summary>Answer</summary>

Slower for each listener, more in all: 139 tokens a second each where one alone gets 669, and 2,020 in all where one gets 657.
</details>

<details><summary>From lesson 15: what would break if positions came from lesson 8's table of places?</summary>

A row could not be moved. Lesson 8 added a place's numbers to the token's at the input, so each key has its place mixed in. Each joining row would have to be read again at its new places.
</details>

Go deeper:

- Kwon and others, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023). The vLLM paper: this batch, with the cache in blocks that rows can share.
- Yu and others, [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (2022). Where joining and leaving between two tokens comes from.
