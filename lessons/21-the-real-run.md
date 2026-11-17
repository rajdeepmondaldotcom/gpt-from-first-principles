# 21. The real run

By the end of this page you have a model that writes short stories, trained on your own machine in an afternoon.

The text is clean and on disk. The size is chosen. The loop can stop and start, and the checks are written. No model has been trained on any of it yet.

## The idea

Nothing on this page is new. Each step is an earlier lesson, used on a run that takes hours and so is hard to do twice.

Check before the run. The script runs two of lesson 13's checks on the untrained model. Its first loss is within 0.15 of ln 4,097 = 8.318, and changing the last token of a window changes no earlier score. Either failure would have cost three hours to find the slow way.

Pick the rate with short runs, as lesson 14 did. Three rates, 264 steps each: 0.0005 ends at 3.292 nats, 0.001 at 3.063 and 0.002 at 3.052. The last two are 0.011 nats apart on one run each, and lesson 14 would not call that a difference. The run needs some rate, and the script takes the lowest loss.

Then the run: 8,921 steps of 32 windows of 256 tokens, which is 73,080,832 tokens, 1.4 times through the training text. Every 500 steps the loop measures the loss on 100,000 unseen tokens and writes a checkpoint. If the machine stops, starting the script again carries on from the last checkpoint. On the graphics chip the resumed run will not match the unbroken one to the last bit, as lesson 10 found.

Read the curve as lesson 13 taught. The two losses fall together. At step 8,500 the training batch scored 1.510 nats and the unseen text 1.567. The model is not storing its text: it has 5 million parameters and 52 million tokens to learn from.

Hold lesson 19's law to its prediction. It said 1.302 nats for this size and budget. The run ends at 1.565, so the law was 0.263 too low. On runs at twice its largest budget it had missed by at most 0.098. This run is 34 times that budget, and the miss is nearly three times larger. We trained one size, so we cannot say whether the law's choice of size was right.

Keep the weights. The script writes them in lesson 16's Llama layout and lesson 10's file format, with the token rows stored once. Lesson 16's loader reads the file back.

## By hand

The score in bits per byte. The run ends at 1.565 nats a token, and an unseen token carries 3.93 bytes.

1.565 / 0.693 = 2.258 bits a token

2.258 / 3.93 = 0.575 bits per byte

The size of the file. 5,374,464 parameters at 4 bytes each is 21,497,856 bytes, about 21.5 MB, plus a short header.

## The code

The two checks, before the first step:

```python
# gpt/story.py
        assert starts_with_no_opinion(model, val_ids[:20_000], SHAPE["vocab"])
        assert reads_no_future(model, val_ids[None, :64])
```

`assert` stops the script with an error when what follows it is false.

The loop is lesson 12's, with a measurement and a checkpoint every 500 steps:

```python
# gpt/story.py
    for step, loss in loop:
        if step % 500 == 0 or step == steps:
            unseen = evaluate(model, val_ids[:100_000])
            notes["curve"].append({"step": step, "train": loss, "val": unseen})
            print(f"step {step}: train {loss:.3f}, unseen {unseen:.3f} nats", flush=True)
            checkpoint(kept, model, optimizer, step, generator)  # a stopped run restarts here
            log.write_text(json.dumps(notes))
```

`notes` holds the rate that was picked and the curve so far. It is written to disk beside the checkpoint, so a restarted run knows both.

## The proof

```bash
uv run pytest tests/test_story.py
```

- Weights saved and loaded back give the same scores, to the last bit.
- The saved file has the names and shapes of lesson 16's reference Llama.

The run itself needs lesson 18's files on disk and about three hours on an M4 laptop's graphics chip. Part of what it prints is below.

```bash
uv run python -m gpt.story
```

```
264 steps at a rate of 0.0005: 3.292 nats
264 steps at a rate of 0.001: 3.063 nats
264 steps at a rate of 0.002: 3.052 nats
parameters: 5,374,464
steps: 8,921, tokens: 73,080,832, rate 0.002
step 500: train 2.451, unseen 2.468 nats
step 1500: train 2.061, unseen 2.029 nats
step 2500: train 1.922, unseen 1.892 nats
step 3500: train 1.807, unseen 1.810 nats
step 4500: train 1.704, unseen 1.743 nats
step 5500: train 1.613, unseen 1.679 nats
step 6500: train 1.500, unseen 1.630 nats
step 7500: train 1.599, unseen 1.590 nats
step 8500: train 1.510, unseen 1.567 nats
unseen stories: 1.565 nats a token, 0.575 bits per byte
```

| Model | Bits per byte on unseen stories |
|---|---|
| Story model, 5,374,464 parameters | 0.575 |

This is a new scoreboard. The stories are simpler text than Shakespeare, so a number here cannot be set beside lesson 16's 2.209.

## What it still cannot do

```
Once upon a time, there was a little girl named Lily. She had a big red bag. Lily liked her bag very much. She wore it every day.
One day, Lily's friend, Sam, came to visit. Sam had a big red bag. Sam said, "I want to pack my bag and have a fun party." Lily was happy. She put all her toys in the bag.
Lily and Sam played games and ate cake. They were very organized. They laughed and had fun. At the end of the day, Lily and Sam were very tired. They went home and had a good night's sleep.
```

The model wrote that after `Once upon a time`. Names stay the same from line to line, the sentences are English, and the story ends. It also wanders: the party arrives from nowhere.

We have not yet said how the model picks each token. The script used a temperature of 0.8 and a top-p of 0.95, and explained neither.

## Check yourself

Predict before you run: the script writes a checkpoint every 500 steps. If the machine stops at step 8,499, how many steps are lost?

<details><summary>Answer</summary>

499. The last checkpoint was written at step 8,000.
</details>

<details><summary>At step 8,500 the training loss is 1.510 nats and the unseen loss 1.567. What would lesson 13 say?</summary>

The gap is 0.057 nats. Lesson 13's model, which stored its text, had a gap of 3.157. This one is still learning what the text has in common.
</details>

<details><summary>The law predicted 1.302 nats and the run gave 1.565. Was the law wrong to use?</summary>

Its loss was too low by 0.263 nats at 34 times its largest budget, so its losses should not be trusted that far out. Whether it chose the size well is a separate question that one run cannot answer.
</details>

<details><summary>From lesson 12: name the four things the checkpoint holds.</summary>

The model's parameters, the optimizer's two averages and its count of steps, the step number, and the state of the random number generator.
</details>

Go deeper:

- Eldan and Li, [TinyStories](https://arxiv.org/abs/2305.07759) (2023). What models of 1 to 30 million parameters learn from this text, with samples from each.
- Andrej Karpathy, [llama2.c](https://github.com/karpathy/llama2.c). The same text and the same layout, with trained models to compare against.
- Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat). One script from tokenizer to chat model, at a budget of about 100 dollars.
