# 33. Prefix caching

By the end of this page a beginning that many prompts share is read once and kept, and each later request starts from where it ends.

Lesson 32's engine reads each prompt from its first token, even when the requests before it began with the same words.

## The idea

Many prompts begin alike. A service puts the same instructions in front of each question. A listener of our storyteller who asks what happens next sends back the whole story so far, with one sentence added.

Lesson 23's cache holds a key and a value for each token read. Lesson 7's mask means that a token's key and value depend only on the tokens before it. So two prompts that agree on their first 100 tokens have the same keys and values for those 100 tokens, whatever comes after. We read them once and keep them, and each later request copies them and reads only what is new.

We keep the cache in pieces of 16 tokens. Each piece is filed under the whole text from the first token to the end of that piece. A new prompt looks up its first 16 tokens, then its first 32, then its first 48, and stops at the first piece that is not there.

1. Reuse stops at the start of the piece that holds the first token that differs. Later tokens may match word for word, but their keys and values were worked out from a different text, and have to be read again.
2. Put what changes last: a prompt that opens with the listener's name or the time of day shares nothing with the next one.
3. At least one token is always read. The store gives back keys and values, and the scores for the first new token take a pass of the model. So a prompt that ends where a stored piece ends reads that last piece again.

The store has a budget in bytes. When full, it throws out the piece used longest ago.

## By hand

A piece of 16 tokens. The story model keeps 6,144 bytes for each token.

16 × 6,144 = 98,304 bytes

Two prompts of 43 and 44 tokens that share their first 40.

| | Tokens | Looked up | Found | Read by the model |
|---|---|---|---|---|
| First prompt | 43 | first 16, nothing there | 0 | 43 |
| Second prompt | 44 | first 16, then first 32 | 32 | 12 |

The first prompt leaves two whole pieces behind, for tokens 1 to 16 and 17 to 32. Its last 11 tokens do not fill a piece. The second prompt shares 40 tokens with it and gets 32 from the store, because pieces come whole.

Now send the first prompt again with its token 21 changed. The first piece still matches. The second is filed under 32 tokens of which one is different, so it is not found: 16 tokens come from the store and 27 are read.

## The code

```python
# gpt/prefix.py
    def fetch(self, ids, blocks):
        """A cache holding the longest stored beginning of `ids`, and how many tokens it covers."""
        found, end = [], self.piece
        while end < len(ids) and tuple(ids[:end]) in self.pieces:  # leave a token to read
            found.append(self.pieces[tuple(ids[:end])])
            end += self.piece
        self.touch(ids, end - self.piece)
        if not found:
            return [{} for _ in range(blocks)], 0
        kept = [
            {name: torch.cat([piece[i][name] for piece in found], dim=2) for name in ("k", "v")}
            for i in range(blocks)
        ]
        return kept, end - self.piece
```

`self.piece` is 16, and `blocks` is the number of blocks in the model. `self.pieces` is a dictionary that remembers the order its entries were last used in. `tuple(ids[:end])` is the text up to a piece's end, in a form a dictionary can file things under. `torch.cat` joins the pieces found into one cache of lesson 23's kind.

```python
# gpt/prefix.py
    def keep(self, ids, kept):
        """Store each whole piece of a prompt's cache that is not stored yet."""
        for end in range(self.piece, len(ids) + 1, self.piece):
            if tuple(ids[:end]) not in self.pieces:
                piece = [
                    {name: one[name][:, :, end - self.piece : end].clone() for name in ("k", "v")}
                    for one in kept
                ]
                self.pieces[tuple(ids[:end])] = piece
                self.used += sum(t.nbytes for one in piece for t in one.values())
        self.touch(ids, len(ids) - len(ids) % self.piece)
        while self.used > self.budget:  # full: throw out the piece used longest ago
            _, piece = self.pieces.popitem(last=False)
            self.used -= sum(t.nbytes for one in piece for t in one.values())
```

Lesson 32's engine calls `fetch` when it takes a request in and `keep` when the request has read its prompt: the two lines of `step` and `admit` that mention a store. `touch` marks a prompt's longest piece as used first, so it is the oldest and goes before the shorter ones it needs.

## The proof

```bash
uv run pytest tests/test_prefix.py
```

- The hand example: 43 tokens read, then 12. Both requests write what they write with no store.
- The first prompt with its token 21 changed reads 27 tokens: one piece is reused, not two.
- With one changing token put first, a prompt of 41 tokens reads all 41.
- A prompt of 32 tokens sent twice reads 16 the second time: its last piece, to get scores to start from.
- A store with room for two pieces is given three. The one used longest ago goes, and a piece that was used again in between stays.
- Given five pieces with room for four, the store keeps no piece without the shorter ones before it.

Predict before you run: 20 prompts share their first 160 tokens and end in 16 tokens of their own. With pieces of 16 tokens, how many tokens does the model read for the second prompt?

```bash
uv run python -m gpt.prefix
```

```
20 prompts that share their first 160 tokens, no store: 4,140 tokens read, first token after 7 ms for half
the same prompts with a store: 1,100 tokens read, first token after 2 ms for half
the answers are the same with and without the store: True
the part that changes put first, with a store: 4,140 tokens read, first token after 7 ms for half
a store with room for 12 pieces: 1,100 tokens read, first token after 2 ms for half
that store holds 1,179,648 bytes of its 1,179,648, in 12 pieces
```

Timings are from an M4 laptop, and they move from run to run.

Each prompt is 176 tokens and each answer 32. With no store the model reads 207 tokens for each request, the prompt and all but the last token of the answer: 4,140 in all. With the store it reads 207 for the first and 47 for each of the other nineteen: 1,100. The wait for the first token falls from 7 ms to 2 ms for half the requests, and the answers do not change.

Put the 16 tokens that change in front, and the store saves nothing: 4,140 tokens again.

A store with room for twelve pieces does as well as one with no limit. Each request uses the ten shared pieces, so they are never the oldest. The pieces that go are the endings no later request asks for.

## What it still cannot do

Twelve pieces in the store take 1.2 MB. The weights take 21.5 MB, each one a 32-bit number, and we have not measured what fewer bits would cost.

## Check yourself

<details><summary>Prediction</summary>

16. The first 160 tokens are ten whole pieces, all in the store after the first prompt. The last 16 are new.
</details>

<details><summary>A service starts each prompt with today's date and then 500 tokens of instructions. What does the store save?</summary>

Nothing across days. Within a day, each whole piece of the date and the instructions. To share across days, put the date after the instructions.
</details>

<details><summary>Why is a piece filed under the whole text up to its end, and not under its own 16 tokens?</summary>

Its keys and values were worked out with the tokens before it in view. The same 16 tokens after a different beginning have different keys and values.
</details>

<details><summary>From lesson 7: if attention had no mask, could two prompts reuse a shared beginning?</summary>

No. Each token would read the tokens after it too, so a beginning's keys and values would change with what follows it.
</details>

Go deeper:

- Zheng and others, [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (2023). Shared beginnings kept in a tree, so that requests which part ways late share the most.
- Kwon and others, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023). Pieces of cache that rows of a batch point to and do not copy.
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). The same idea as a service sells it, with the rule about putting what changes last.
