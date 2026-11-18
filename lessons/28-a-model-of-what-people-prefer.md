# 28. A model of what people prefer

By the end of this page a second model has learned to score stories from pairs in which a reader liked one more, and you have watched what happens to the readers' number as the storyteller is trained against that score.

Lesson 27's score gave nothing for an ending, and 59 of 200 stories now never reach one, where 4 did. A reader would mark that down, and that score cannot.

## The idea

In some tasks, Christiano and others found a choice between two easier to give than a number. So the raw material is pairs: two stories written to one order, and which of them a reader preferred.

We have no readers, so a short function stands in for them. It gives each story a number: 0 unless the story ends, runs to 40 words or more and says no sentence twice, and otherwise lesson 27's score. Of two stories, the readers prefer the one with the higher number. The model on this page never sees that rule, only pairs. Knowing the rule, we can set a learned score beside it, which a real project cannot.

The scorer is a model that reads an order and its story and gives back one number. Its body is a copy of the storyteller, lesson 25's model, which already reads stories, with one new layer on top that turns the stream at the last token into a single number.

For each pair it scores both stories, and the difference between the scorer's two numbers is a bet on which the reader chose. When the first story leads by x, the bet puts a chance of 1 / (1 + e^-x) on it: one half at a lead of 0, and nearer 1 the larger the lead. The loss is lesson 1's surprise at the reader's real choice. Only differences between scores enter the bet, so its numbers have no fixed zero.

A trained scorer can stand where lesson 27's score stood: the storyteller writes stories, the scorer scores them, and the step raises the chance of those above their group's average. We call that training the push.

The scorer learned from stories the storyteller wrote before any of this training. Training now pushes the storyteller toward whatever the scorer rates highest, and nothing makes its number hold for stories unlike the ones it learned from. So we watch the scorer's number, which the training sees, and two it does not: the readers', and lesson 27's score. Ouyang and others add a penalty for drifting from the starting model. The push had none, and lesson 29's loss builds one in.

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
- A toy taste: of two random texts, the one with more 3s is better. From 600 pairs a small scorer ranks over 9 in 10 of 200 unseen pairs right, from about half.
- The readers' three verdicts above, and two more: a whole story with all four gets 1, and a story of 7 words gets 0.
- From four stories for one order, a good one, a fair one and two bad ones, five pairs are made, each with the better story first. The two bad ones tie and make no pair. An order held back keeps its pairs out of the learning.

The stories are the 1,600 that lesson 27's first way wrote, four for each of 400 orders, all by lesson 25's model. Each two stories for one order make a pair, unless the readers give both the same number. The push is 100 rounds of 16 stories at a rate of 0.0003, on orders new to lesson 27.

Predict before you run: as the storyteller is trained against the scorer, the scorer's number for its stories will rise. What will the readers' number do?

```bash
uv run python -m gpt.scorer
```

```
pairs to learn from: 1,386, from 340 orders. Pairs held back: 240, from 60 orders
step 1: 0.916 on 16 pairs. ln 2 is 0.693
step 100: 0.562 on 16 pairs. ln 2 is 0.693
step 200: 0.196 on 16 pairs. ln 2 is 0.693
step 300: 0.024 on 16 pairs. ln 2 is 0.693
held-back pairs ranked right before any training: 109 of 240
after: 140 of 240, 95% interval 0.51 to 0.65
before the push, the 240 stories held back: the scorer gives them +3.41 on average, the readers 0.58, lesson 27's score 0.61
rounds 1 to 20: the scorer gives the stories +4.85 on average, the readers 0.51, lesson 27's score 0.55
rounds 21 to 40: the scorer gives the stories +8.60 on average, the readers 0.31, lesson 27's score 0.40
rounds 41 to 60: the scorer gives the stories +11.43 on average, the readers 0.20, lesson 27's score 0.23
rounds 61 to 80: the scorer gives the stories +12.41 on average, the readers 0.16, lesson 27's score 0.17
rounds 81 to 100: the scorer gives the stories +12.72 on average, the readers 0.05, lesson 27's score 0.09
the last order asked for: Bob, movie, movies, watch
on 200 unseen orders the readers give its stories 0.03, where they gave 0.57. The change has a 95% interval of -0.58 to -0.51
```

The scorer's loss on a batch of 16 of its own pairs falls from 0.916 to 0.024. Of the 240 held-back pairs it ranked 109 right before training and 140 after: 0.51 to 0.65 against a coin's 0.5, drawn order by order, since one order's pairs share stories.

The scorer's number rose from +3.41 to +12.72. On 200 orders the push did not train on, the readers gave its stories 0.57 before it and give 0.03 after, a change of -0.58 to -0.51.

## What it still cannot do

```
One day, a happy sad girl named Mia wanted to find more. She wanted to bake another again. She asked her friend Tom, help her teach D sho home. She wanted to eat more.


 Bessie wanted to teach Bella please forgive her friend. They both wanted to find more. They wanted to marry one again.
 could not plan, they asked their friend, Bunny, for help. They wanted to marry, again. They offered they patient and said, "Here". They fit bests for their home. They agreed and ate the chocolate.
Bby was happy again. They were happy again. They could not sell past again. They had fun and sang their new things again. They were happy again. They were happy again because they did their choory together.
```

The last order asked for Bob, movie, movies and watch, and the story above is what the storyteller now writes. The scorer is a second model to train and keep, and training against it drove the readers' number to 0.05. The pairs already say which story is better, and we have not learned from them directly.

## Check yourself

<details><summary>Prediction</summary>

Fall: from 0.58 to 0.05 over the push, and from 0.57 to 0.03 on unseen orders.
</details>

<details><summary>Why can training against the scorer make stories worse, when the scorer ranks most unseen pairs right?</summary>

It learned from lesson 25's stories only. Here the number it gives rose while the readers' fell, so its number stopped tracking theirs.
</details>

<details><summary>From lesson 13: the scorer's loss on a batch of its own pairs fell to 0.024, and it ranks 140 of 240 unseen pairs right. What has happened, and what would you try?</summary>

It has stored its pairs, as lesson 13's model stored its text. Lesson 13 measured dropout and weight decay together as a cure. More pairs is another to try.
</details>

Go deeper:

- Christiano and others, [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (2017). A scorer learned from people's choices between pairs, and a model trained against it.
- Ouyang and others, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022). The same idea applied to a large language model.
