# 28. A model of what people prefer

By the end of this page a second model has learned to score stories from pairs in which a reader liked one more, and you have watched what happens to the storyteller when it is trained hard against that score.

Lesson 27's check was a program: four words, found or not found. Most of what people want from a story has no such program. People can still say which of two stories they like better.

## The idea

People are poor at giving a story a number and good at choosing between two. So the raw material is pairs: two stories written to one order, and which of them a reader preferred.

We have no readers, so a short function stands in for them, and we say now what it likes. It gives each story a number: 0 unless the story ends, runs to 40 words or more and says no sentence twice, and otherwise lesson 27's score. Of two stories, the readers prefer the one with the higher number. The model on this page never sees that rule. It sees pairs. Having the rule lets us do what a real project cannot: set a learned score beside the truth.

The scorer is a model that reads an order and its story and gives back one number. Its body is a copy of the story model, which can read stories already. On top goes one new layer that turns the stream at the last token into a single number.

It learns from a pair like this. It scores both stories. Treat the difference between the two scores as a bet on which the reader chose. When the first story leads by x, the bet puts a chance of 1 / (1 + e^-x) on it: one half at a lead of 0, and nearer 1 the larger the lead. The loss is the surprise of lesson 1 at the reader's real choice.

A trained scorer can stand where lesson 27's check stood. The storyteller writes stories, the scorer scores them, and the step raises the chance of the stories that scored above their group's average.

Here is the trap. The scorer learned from stories the storyteller wrote before any of this training. Training now pushes the storyteller toward whatever the scorer rates highest, and that can be a story unlike any the scorer has seen, where its number means little. The storyteller is paid to find the scorer's mistakes. So we watch two numbers through the training: the scorer's, which the training sees, and the readers', which it does not.

## By hand

The loss for one pair. The scorer gives the better story 3.0 and the worse one 1.0.

| Step | Sum | Value |
|---|---|---|
| Lead of the better story | 3.0 - 1.0 | 2.0 |
| Chance the bet puts on it | 1 / (1 + e^-2.0) | 0.881 |
| Loss | -ln 0.881 | 0.127 |

With equal scores the lead is 0, the chance is 0.5 and the loss is ln 2 = 0.693.

The readers. An order asks for Mira, cat, moon and brave.

| Story | The readers give it |
|---|---|
| Ends, 60 words, has Mira, cat and brave | 0.75 |
| The same, with one sentence said twice | 0 |
| Has all four, and is cut off before its end | 0 |

## The code

```python
# gpt/scorer.py
class Scorer(nn.Module):
    """A GPT's body with one new layer on top. It reads a text and gives it a single number."""

    def __init__(self, body):
        super().__init__()
        self.body = body
        self.head = nn.Linear(body.embed.weight.size(1), 1)

    def forward(self, ids, last):
        """ids: (batch, tokens), padded. last: the place of each row's final real token."""
        x = self.body.embed(ids)
        for block in self.body.blocks:
            x = block(x)
        read = self.body.norm(x)[torch.arange(len(ids)), last]  # that place has read the whole text
        return self.head(read).squeeze(-1)
```

`body.embed.weight.size(1)` is the model's width, and `squeeze(-1)` turns a column of single numbers into a row. Texts in a batch have different lengths, so each is padded at its end, and `last` says where each one stops. Padding after that place cannot reach back to it: lesson 7's mask.

```python
# gpt/scorer.py
def comparison_loss(better, worse):
    """Low when the better text of each pair scores higher. ln 2 when the scorer cannot tell."""
    return -nn.functional.logsigmoid(better - worse).mean()
```

`logsigmoid(x)` is the log of 1 / (1 + e^-x), worked out so that large numbers do not overflow.

## The proof

```bash
uv run pytest tests/test_scorer.py
```

- Padding keeps each text's last place, and does not change a text's score.
- The loss is ln 2 for equal scores and 0.127 for the hand example.
- A toy taste: of two random texts, the one with more 3s is better. From 600 pairs a small scorer ranks 200 unseen pairs more than 9 in 10 right, where it got about half before training.
- The readers' three verdicts above, and two more: a whole story with all four gets 1, and a story of 7 words gets 0.
- From four stories for one order, a good one, a fair one and two bad ones, five pairs are made, each with the better story first. The two bad ones tie and make no pair.

The stories are the 1,600 that lesson 27's first way wrote, four for each of 400 orders, all by lesson 25's model as it was. Each two stories for one order make a pair, unless the readers give both the same number.

```bash
uv run python -m gpt.scorer
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

The scorer is a second model to train, to keep and to be fooled. The pairs themselves already say which story is better, and we have not learned from them directly.

## Check yourself

Predict before you run: as the storyteller is trained against the scorer, the scorer's number for its stories will rise. What will the readers' number do?

<details><summary>Answer</summary>

@@PREDICT@@
</details>

<details><summary>Why can training against the scorer make stories worse, when the scorer ranks most unseen pairs correctly?</summary>

It ranks stories like the ones it learned from. Training moves the storyteller away from those, toward whatever earns a high number, and far from its lessons the scorer's number is a guess.
</details>

<details><summary>From lesson 13: the scorer's loss on its own pairs fell to 0.004, and it ranks 137 of 210 unseen pairs right. What has happened, and what would you try?</summary>

It has stored its pairs, as lesson 13's model stored its text. That lesson's cures apply: more pairs, dropout or weight decay, or stopping when the unseen pairs stop getting better.
</details>

Go deeper:

- Christiano and others, [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (2017). A scorer learned from people's choices between pairs, and a model trained against it.
- Ouyang and others, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022). The same idea applied to a large language model.
- Lambert, [the RLHF book](https://rlhfbook.com/). A whole book on this page and the two beside it.
