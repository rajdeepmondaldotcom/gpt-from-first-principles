# 35. Speculative decoding

By the end of this page a small model guesses four tokens ahead, the big model checks all four in one pass, and each token still comes with the chance the big model alone would give it.

The model is smaller now. It still runs once, top to bottom, for each token it writes.

## The idea

One pass of the model gives a row of scores at each place it reads, as training has used since lesson 10. So if something could guess the next four tokens, the big model could check all four guesses, and add a fifth token, for the price of one pass.

A small model does the guessing. We call it the draft. It writes four tokens, one at a time. Then the big model reads all four in one pass.

The obvious check keeps a guess when it is the big model's top token. That writes only what greedy writes, and lesson 22 showed why we want to draw by chance. The check below keeps the big model's chances exactly.

Say the draft drew a token to which it gave the chance q, and the big model gives that token the chance p.

- If p is at least q, keep the token.
- If p is less than q, keep it with a chance of p / q.
- If it is thrown out, draw a token from what the big model has left over: p - q for each token where that is above zero, shared out to sum to 1. Drop the guesses after it too, since they followed a token we did not keep.
- If all four are kept, the same pass has given the big model's chances for a fifth token. Draw it.

Why this is exact: a token is written either because it was drafted and kept, or because a guess was thrown out and it was drawn from the leftover. The hand example adds the two for each token and gets p, and so does a test.

So the draft cannot change what is written, only how many passes it takes: five tokens a pass at best, and one at worst.

Which draft agrees most? We train two of one size. The first learns from the stories. The second learns to copy the big model. Its loss is lesson 3's, with the big model's whole row of chances as the target where the text gave one right token: each token's surprise, weighted by the chance the big model gives that token. This is called distillation.

Whether it is faster is a measurement, and lesson 34 is a warning: a smaller model there wrote slower. A round costs four passes of the draft and one of the big model.

## By hand

Three tokens. The draft gives them chances q of 0.2, 0.6 and 0.2. The big model gives them p of 0.5, 0.3 and 0.2.

| Token | q | p | Keep chance | Drafted and kept | Left over, p - q |
|---|---|---|---|---|---|
| First | 0.2 | 0.5 | 1 | 0.2 | 0.3 |
| Second | 0.6 | 0.3 | 0.5 | 0.3 | 0 |
| Third | 0.2 | 0.2 | 1 | 0.2 | 0 |

A guess is kept 0.7 of the time. The leftover is all on the first token, so a redraw always gives the first. The first token is then written 0.2 + 0.3 = 0.5 of the time, the second 0.3 and the third 0.2: the big model's chances.

Tokens for each pass. A round always writes one token, a second if the first guess was kept, a third if two were, and so on. If each guess is kept with a chance of a, that is 1 + a + a² + a³ + a⁴ tokens on average: 3.36 at a = 0.8, and 1.94 at a = 0.5.

## The code

```python
# gpt/speculate.py
def keep_chance(p, q):
    """The chance of keeping each token the draft might pick: 1 where the big model likes it more."""
    return (p / q).clamp(max=1)


def leftover(p, q):
    """What to draw from after a rejection: the chance the big model has and the draft lacks."""
    left = (p - q).clamp(min=0)
    return left / left.sum().clamp(min=1e-12)
```

`clamp(max=1)` cuts each number above 1 down to 1, and `clamp(min=0)` lifts each number below 0 up to 0.

```python
# gpt/speculate.py
        for j, guess in enumerate(guesses):
            if torch.rand((), generator=generator) < keep_chance(p[j], q[j])[guess]:
                ids.append(guess)
                counts["kept"] += 1
            else:  # the first rejection ends the round: draw this token again, drop the rest
                ids.append(torch.multinomial(leftover(p[j], q[j]), 1, generator=generator).item())
                break
        else:  # every guess kept: the big model's last row gives one more token for nothing
            ids.append(torch.multinomial(p[k], 1, generator=generator).item())
```

`p[j]` and `q[j]` are the two models' rows of chances for guess `j`, and `k` is the number of guesses, 4. `torch.rand(())` is one number between 0 and 1, drawn evenly, so it falls under the keep chance that share of the time. A `for` loop's `else` runs when the loop was not broken out of. Both models keep lesson 23's cache, and after a rejection each cache forgets the tokens that were dropped.

## The proof

```bash
uv run pytest tests/test_speculate.py
```

- The hand example: keep chances of 1, 0.5 and 1, and a leftover that is all on the first token.
- For 50 random pairs of p and q over 8 tokens, drafted and kept plus redrawn equals p for each token.
- Two tokens written 3,000 times with a draft that disagrees with the big model: each of the 25 pairs comes within 0.03 of the chance the big model alone gives it.
- With the big model as its own draft, each guess is kept and 20 tokens take 4 passes.
- A poor draft has guesses thrown out, and never needs more than one pass a token.
- The copying loss equals lesson 3's loss when the big model is sure of one token, and is lowest when the draft's chances equal the big model's.

The script trains the two drafts on the graphics chip, then times the writing on the processor.

```bash
uv run python -m gpt.speculate
```

```
@@OUTPUT@@
```

Timings are from an M4 laptop that was running other jobs, and they move from run to run.

@@RESULTS@@

## What it still cannot do

Each script so far runs once, in a terminal, and stops. Nobody else can ask our model for a story.

## Check yourself

Predict before you run: the draft is the big model itself. How many tokens does each pass of the big model give, with four guesses a round?

<details><summary>Answer</summary>

Five. p equals q, so each guess is kept, and the pass that checked them gives a fifth token.
</details>

<details><summary>From lesson 22: why not keep a guess only when it is the big model's top token?</summary>

That is greedy: the same text each time. Drawing by chance is what gave twenty different stories, and this check keeps each token's chance as the big model has it.
</details>

Go deeper:

- Leviathan and others, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022). The keep rule, and the sum for tokens for each pass.
- Chen and others, [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (2023). The same rule, found independently, with timings on a large model.
- Hinton and others, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015). Training a small model on a large model's chances.
