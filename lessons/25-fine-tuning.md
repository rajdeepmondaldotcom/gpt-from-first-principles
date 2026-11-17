# 25. Fine-tuning

By the end of this page the story model takes an order, a name and three words, and you have measured how often it obeys, whether it stops by itself, and what the new skill cost the old one.

Asked in three lines for a name and three words, the model passes 0 of 100. It was trained to continue stories, and no story in its text begins with a line that says what to write.

## The idea

Our model writes what its training text taught it to write. So we make text in which an order is followed by a story that obeys it, and train on that. Training a trained model further on new text is called fine-tuning.

The examples come from the function that made lesson 24's tasks. It reads a story and writes down its hero's name and three of its less common words. Run over training stories, it makes examples: the three lines of an order, then the story they came from, which obeys them already. The first 20 million training tokens give 60,404.

1. Only the answer is scored. An example is a prompt and then an answer. A loss on each token would mark the model on guessing an order's name and words, which it cannot know. So the loss counts only the places whose target is part of the answer.
2. Each answer ends with the end token, and that token is scored. The model learns that after a story comes the end, and lesson 22's writing loop stops there.
3. Training starts from lesson 21's weights. It runs 2,000 steps of 32 examples at a peak rate of 0.001, half of lesson 21's.

Examples have different lengths and a batch is a grid, so each example is padded at its end. Padded places are not scored, so any token will do, and we use the end token. The hand example pads to its longest row. The run pads each batch to the model's whole context: in our first try on this Mac's graphics chip, training stalled when the shape of the batch kept changing.

We measure three things before and after, on 200 tasks made from unseen stories: how many pass lesson 24's exact check, what share of the three asked words turn up, and how many stories reach the end token by themselves. The share is the finer measure: a story with two of the three words fails the check.

The new skill may cost the old one. It trains each weight, and the examples hold stories only after an order. So we score 500 plain unseen stories before and after, take the difference at each token, and give the mean difference lesson 24's interval, drawn story by story.

## By hand

A toy prompt is the three tokens `1 2 |`, and its answer is `1 2` and then the end token.

| Place | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| The model reads | 1 | 2 | `\|` | 1 | 2 |
| It should write next | 2 | `\|` | 1 | 2 | end |
| Scored | no | no | yes | yes | yes |

The first scored place is 2: the model has read the whole prompt and should write the answer's first token. Places 0 and 1 ask it to write the prompt, and are left out.

A second example has a prompt of 2 tokens, an answer of 1 and the end token: 3 places. In a batch with the first, it gets 2 padded places at its end, neither scored.

## The code

```python
# gpt/finetune.py
def collate(examples, pad, device="cpu", width=None):
    """Stack examples of different lengths. `scored` marks the places whose target is answer."""
    width = width or max(len(ids) for ids, _ in examples) - 1  # as wide as the longest needs
    x = torch.full((len(examples), width), pad, device=device)
    y, scored = x.clone(), torch.zeros_like(x, dtype=torch.bool)
    for row, (ids, start) in enumerate(examples):
        ids = torch.tensor(ids, device=device)
        x[row, : len(ids) - 1], y[row, : len(ids) - 1] = ids[:-1], ids[1:]
        # place start - 1 predicts the answer's first token
        scored[row, start - 1 : len(ids) - 1] = True
    return x, y, scored


def answer_loss(model, x, y, scored):
    """Lesson 10's loss, counted only where the target is part of an answer."""
    return cross_entropy(model(x)[scored], y[scored])
```

An example is a pair: the token ids of prompt, answer and end token, and `start`, the place where the answer begins. An example of n tokens fills n - 1 places, since its last token has no next one. `torch.full` makes a grid filled with one value, here the padding. `x.clone()` is a copy, and `zeros_like` a grid of the same shape filled with False. `model(x)[scored]` keeps the rows of scores at the places marked true, as one list.

## The proof

```bash
uv run pytest tests/test_finetune.py
```

- The hand example, and a second one padded beside it, give the grids above for both rows.
- An example ends with the end token, and `tune` pads each batch to the model's context.
- Changing a target inside a prompt, or in the padding, leaves the loss as it was. Changing a target inside an answer moves it.
- A small model learns a toy task in 400 steps: after two tokens and a separator, write the same two tokens and stop. Its loss falls from above 2 to below 0.1, and it stops by itself, which it did not before.

The script trains on the graphics chip and measures on the processor.

```bash
uv run python -m gpt.finetune
```

```
examples to learn from: 60,404
before: 0 of 200 tasks passed, 95% interval 0.00 to 0.02
before: 0.14 of the asked words used, 95% interval 0.11 to 0.17
before: 177 stories end by themselves, plain stories 1.480 nats a token
step 250: 1.344 nats on the answers
step 500: 1.291 nats on the answers
step 750: 1.325 nats on the answers
step 1000: 1.261 nats on the answers
step 1250: 1.222 nats on the answers
step 1500: 1.236 nats on the answers
step 1750: 1.256 nats on the answers
step 2000: 1.116 nats on the answers
after: 27 of 200 tasks passed, 95% interval 0.09 to 0.19
after: 0.56 of the asked words used, 95% interval 0.52 to 0.60
after: 195 stories end by themselves, plain stories 1.549 nats a token
plain stories, after minus before: +0.069 nats, 95% interval +0.059 to +0.079
```

Each step's line is the loss of one batch of 32, so it rises and falls. Of 200 tasks the model passed 0 before fine-tuning and passes 27 after, and the interval of 0.09 to 0.19 is clear of the 0.02 that bounded the rate before. The finer measure moves further: stories use 0.56 of the asked words, up from 0.14.

195 of 200 stories now reach the end token, where 177 did.

The price is on plain stories: 1.549 nats where they scored 1.480, a difference of +0.069 with an interval of +0.059 to +0.079.

## What it still cannot do

```
Mira and Joe were best friends. They played together every day. One day, Mira wanted to take an adventure to the moon. Joe wanted to help. He turned on the light and looked around. 
“Where are you going?” Joe asked.
“I’m going to the moon,” Mira said. “It will be a very much better.”
“What will you like?”
So Mira took off her shoes and then meowed. Joe was happy. He now had to wipe the moon. 
Tina said goodbye, “That was the most amazing.”
Joe smiled. He knew that to make the moon look beautiful, but it was still a few stars in the night sky.
```

From this lesson on, the sample answers an order: here Mira, cat, moon and brave. The story has Mira and the moon, no cat and nothing brave, and fails. Across the 200 stories, 0.44 of the asked words go missing.

The fine-tune trained all 5,374,464 of the model's weights, and the result is a second file of 21.5 MB. Ten tasks would mean ten such files.

## Check yourself

Predict before you run: before fine-tuning, the model has never seen an order. What share of the asked words will its stories use?

<details><summary>Answer</summary>

About one in seven: 0.14. The asked words are ordinary story words, and some turn up by chance. No story of the 200 has the name and all three.
</details>

<details><summary>Why is the prompt left out of the loss?</summary>

Scoring it would train the model to write orders, and mark it down for not guessing a name and three words it cannot know.
</details>

<details><summary>From lesson 24: why take the difference on plain stories at each token, and not compare two means?</summary>

Both models read the same 500 stories, so most of what moves a story's loss is the same for both and cancels in the difference. Lesson 14 did the same with seeds: each gave one difference between two models that had the same luck.
</details>

Go deeper:

- Eldan and Li, [TinyStories](https://arxiv.org/abs/2305.07759) (2023). It also trains small models to follow orders of this kind.
- Howard and Ruder, [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (2018). An early paper on fine-tuning a trained language model for a task.
- Raschka, [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch), chapter 7. Fine-tuning a GPT-2 to follow written instructions.
