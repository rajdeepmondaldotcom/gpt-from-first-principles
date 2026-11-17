# 22. Sampling

By the end of this page you control how the model picks each token: the same story each time, or a new one on each run.

Lesson 21's model wrote a story after `Once upon a time`. The script that asked for it set a temperature of 0.8 and a top-p of 0.95, and explained neither.

## The idea

The model does not write. It gives a chance to each of 4,097 tokens, and something outside the model has to pick one, add it to the text and ask again. How we pick is a choice we make after training, and it changes what we read.

The plainest rule takes the token with the top score each time. It is called greedy. It uses no chance, so the same prompt gives the same story on every run.

The rule that matches training draws a token by its chance, as lesson 1 did. A token with a chance of 0.3 comes three times in ten. Each run now gives a different story. It also means that a token with a chance of one in a thousand comes once in a thousand draws, and a story has hundreds of draws.

Temperature sits between the two. Divide each score by a number T before the softmax. With T under 1 the gaps between scores grow, so likely tokens become more likely. With T over 1 the gaps shrink and the chances move toward even. A T of 0.5 squares each chance and then shares the total out again. As T goes to 0 the top token takes all of it, and we treat a T of 0 as greedy.

Top-p goes after the unlikely tokens. Sort the tokens from most to least likely and keep the smallest set whose chances add up to p. Drop the rest and share their chance among the kept ones. With a p of 0.9, the unlikely tenth is never drawn.

Writing stops when the model picks the end token, or when it has written as many tokens as we allow.

## By hand

Four tokens with chances 0.5, 0.3, 0.15 and 0.05.

| | First | Second | Third | Fourth |
|---|---|---|---|---|
| Chance | 0.5 | 0.3 | 0.15 | 0.05 |
| Squared, for T = 0.5 | 0.25 | 0.09 | 0.0225 | 0.0025 |
| Divided by their total, 0.365 | 0.685 | 0.247 | 0.062 | 0.007 |

Now top-p at 0.8 on the first row. The first token alone has 0.5, which is under 0.8. The first two have 0.8, which reaches it. So two are kept, and their chances become 0.5 / 0.8 = 0.625 and 0.3 / 0.8 = 0.375. At a p of 0.81 the third would be kept too.

## The code

```python
# gpt/generate.py
def choose(scores, temperature=1.0, top_p=1.0, generator=None):
    """Pick the next token from one row of scores."""
    if temperature == 0:
        return scores.argmax().item()  # no chance involved: the top score wins
    chances = softmax(scores / temperature)  # under 1 sharpens the row, over 1 flattens it
    order = chances.argsort(descending=True)
    before = chances[order].cumsum(0) - chances[order]  # the total chance of the likelier tokens
    chances[order[before >= top_p]] = 0  # the likelier ones already reach top_p: drop this one
    return torch.multinomial(chances.cpu(), 1, generator=generator).item()
```

`argmax` is the place of the largest score. `argsort(descending=True)` gives the places of the chances from largest to smallest. `cumsum` is a running total, so `before` holds, for each token, the chance of all the tokens ahead of it. `torch.multinomial` draws by weight and does not need the weights to add up to 1.

## The proof

```bash
uv run pytest tests/test_generate.py
```

- A temperature of 0 picks the top score every time.
- At a temperature of 1, each of the four tokens comes as often as its chance, over 20,000 draws.
- A temperature of 0.5 gives the hand example's 0.685, 0.247, 0.062 and 0.007. A temperature of 100 gives nearly even chances.
- Top-p keeps 1 token at 0.5, 2 at 0.6 and at 0.8, 3 at 0.81 and 4 at 1.
- At a top-p of 0.8 the two kept tokens come at 0.625 and 0.375.
- Choosing leaves the scores as they were.
- Writing stops at the stop token.

The script writes 20 stories from `Once upon a time` at each of five settings. It counts how many of the 20 differ, how many reach their end inside 250 tokens, and how many of their words the training text never uses.

```bash
uv run python -m gpt.generate
```

```
temperature 0: 1 different stories of 20, 20 reach their end, 0.0% of their words made up
  'Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she went to the park with her mom '
temperature 0.5: 20 different stories of 20, 19 reach their end, 0.0% of their words made up
  'Once upon a time, there was a little girl named Amy. Amy loved to eat spaghetti. One day, she saw a big, round thing in the kitchen. It looked like a '
temperature 1: 20 different stories of 20, 20 reach their end, 0.3% of their words made up
  'Once upon a time, in a big forest, there was a small cabin. In the cabin lived a funny bear named Bobo. Bobo had a favorite seat in the woods. He was '
temperature 1.5: 20 different stories of 20, 18 reach their end, 13.2% of their words made up
  'Once upon a time, there was an apron. The apron breatamed purple lying coiled around shining in the shadows across the wasops until there was just as '
temperature 1, top-p 0.9: 20 different stories of 20, 20 reach their end, 0.0% of their words made up
  'Once upon a time, there was a bird. The bird had a pretty feather. The bird liked to fly high in the sky. The bird was a happy bird.\nOne day, the bird'
```

A temperature of 0 writes one story twenty times. At 0.5 and at 1 the twenty differ, and nearly every word is one the training text uses. At 1.5 the stories fall apart: 13.2% of their words are made up, such as `breatamed`. Top-p at 0.9 keeps the variety of a temperature of 1, and in these twenty stories no word is made up.

## What it still cannot do

To write each new token, the model reads the whole text again from its first token. A story of 200 tokens is read 200 times, each time one token longer.

## Check yourself

Predict before you run: at a temperature of 0, how many of the 20 stories differ?

<details><summary>Answer</summary>

One. Nothing is drawn by chance, so each run takes the same path.
</details>

<details><summary>Chances of 0.6, 0.3 and 0.1, with a top-p of 0.5. What are the chances after it?</summary>

1, 0 and 0. The first token alone reaches 0.5, so it is the only one kept.
</details>

<details><summary>Chances of 0.8 and 0.2 at a temperature of 0.5. What do they become?</summary>

The squares are 0.64 and 0.04, which total 0.68. So 0.941 and 0.059.
</details>

<details><summary>From lesson 1: which setting draws each token by the chance the model gave it?</summary>

A temperature of 1 with a top-p of 1. That is lesson 1's rule, and the loss in training measures the model under that rule.
</details>

Go deeper:

- Holtzman and others, [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (2019). Where top-p comes from, and what goes wrong without it.
- Hugging Face, [How to generate text](https://huggingface.co/blog/how-to-generate). The same rules and a few more, with examples.
