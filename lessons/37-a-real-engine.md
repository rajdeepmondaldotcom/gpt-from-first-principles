# 37. A real engine

By the end of this page our model runs in Apple's MLX engine, gives the scores it gives in ours, and you have timed both engines on both of this Mac's chips.

Lesson 31 measured 539 tokens a second for one listener where memory alone allows 4,231, and lessons 34 and 35 made it no faster. We have not seen what an engine built for speed does with the same model.

## The idea

Our model is one file of weights and about 200 lines of Python. An engine is another program that reads the same file and does the same arithmetic, built for speed. Handing a model over takes two things: the engine must find the weights, and do with them what we do.

Finding the weights is settled: in lesson 21 we saved them under the names that engines for Llama-format models look for, in lesson 10's file format. The engine still needs the sizes, and the three settings our code fixes (lesson 16): the width inside SwiGLU, the small number inside RMSNorm and the base of the rotary speeds. Those go in a small text file, `config.json`. The export copies the weights and writes that file.

Doing what we do is harder to see. The engine has its own code for attention, rotary positions, RMSNorm and SwiGLU. If one detail differs, such as which entries lesson 15's rotation pairs up, its scores differ from ours with no error shown. So we feed both engines the same tokens and compare each score. We do not expect the last bits to agree: an engine may add the same numbers in another order, and 32-bit sums depend on the order.

Then speed. Both engines write 128 tokens after the same 64-token prompt and take the top token each time, and we keep the fastest of three tries. We time both engines on the processor and on the graphics chip, and each with sixteen at once, ours on the processor only.

One token's pass makes about 500 calls into PyTorch, and each call costs some time whatever the size of its arithmetic. If that fixed cost is a large part of a pass, a far smaller model will not be much faster. So the script also times an untrained model as deep as ours and an eighth as wide: 26 times fewer weights, the same calls.

## By hand

The story model's sizes, as the config file gives them.

| Name in the file | Value | Where it comes from |
|---|---|---|
| `hidden_size` | 256 | the width |
| `num_hidden_layers` | 6 | the blocks |
| `num_attention_heads` | 8 | 256 / 32 |
| `num_key_value_heads` | 4 | grouped heads |
| `max_position_embeddings` | 256 | the context |
| `intermediate_size` | 682 | 8 × 256 / 3, rounded down |
| `vocab_size` | 4,097 | 4,096 and the end token |

Where a pass's time goes. Say a pass costs a fixed part F for its calls and a part A that grows with the count of weights. In the run below our model takes 1,000 / 775 = 1.29 ms a token, and the model with 26 times fewer weights and the same calls takes 1,000 / 1,648 = 0.61 ms.

1.29 = F + A

0.61 = F + A / 26

Subtract: 0.68 = A × 25 / 26, so A is 0.71 ms and F is 0.58 ms.

## The code

```python
# gpt/export.py
def export(folder, out):
    """Make a folder that engines for Llama-format models can load: our weights file as it is,
    and a config file that names each size the way those engines expect."""
    out.mkdir(parents=True, exist_ok=True)
    shape = json.loads((folder / "shape.json").read_text())
    shutil.copy(folder / "model.safetensors", out / "model.safetensors")
    wide = read_safetensors(folder / "model.safetensors")["model.layers.0.mlp.gate_proj.weight"]
    config = {
        "model_type": "llama",
        "vocab_size": shape["vocab"],
        "hidden_size": shape["width"],
        "num_hidden_layers": shape["layers"],
        "num_attention_heads": shape["heads"],
        "num_key_value_heads": shape["kv_heads"] or shape["heads"],
        "max_position_embeddings": shape["context"],
        "tie_word_embeddings": True,
        "intermediate_size": wide.size(0),  # the three settings our code fixes (lesson 16)
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
```

`shape.json` is lesson 21's file of sizes, and `read_safetensors` lesson 10's reader. `gate_proj` is the Llama name for SwiGLU's first grid, stored with a row for each output, so its count of rows is SwiGLU's width. A `kv_heads` of None means a key head for each head. `tie_word_embeddings` says the token rows and the scores layer are one grid, as since lesson 10.

## The proof

```bash
uv run pytest tests/test_export.py
```

- The config file of a small model names each of its sizes, and the weights file is copied byte for byte.

One more test needs Apple's engine, which exists for Macs. As in lesson 5, `--extra reference` installs it and `-m reference` runs its tests:

```bash
uv run --extra reference pytest -m reference tests/test_export.py
```

- A small model with random weights and grouped heads, loaded in MLX, gives our scores to 0.0001, and the same top token at each place.

```bash
uv run python -m gpt.export
```

```
256 unseen tokens, largest gap between the two engines' scores: 0.000013
places where both engines put the same token on top: 256 of 256
one token's pass makes 516 calls into PyTorch
a model 26 times smaller makes 516
ours, processor: 775 tokens a second
ours, processor, 16 at once: 4,134 tokens a second
ours, processor, a model 26 times smaller: 1,648 tokens a second
ours, graphics chip: 104 tokens a second
Apple's engine, graphics chip: 298 tokens a second
Apple's engine, graphics chip, 16 at once: 5,921 tokens a second
Apple's engine, processor: 942 tokens a second
Apple's engine, processor, 16 at once: 5,943 tokens a second
the two engines write the same first 128 tokens of 128
```

Timings are from an M4 laptop, and they move from run to run. The chip's three timings are the least settled: an earlier run read two to four times higher on each, and we have not found why.

The engines agree: the largest gap between their scores is 0.000013, the top token is the same at 256 of 256 places, and they write the same 128 tokens.

On the processor, one listener gets 775 tokens a second from our engine and 942 from Apple's. Lesson 31's 539 drew by chance after a prompt three times as long. With 16 at once it is 4,134 against 5,943. On the graphics chip ours writes 104, about a seventh of its speed on the processor. Apple's writes 298 there alone and 5,921 with 16 at once, level with its processor.

With the small model we answer part of lesson 31's question. It makes the same 516 calls on 26 times fewer weights and writes 1,648 tokens a second, 2.1 times as fast. By the sum above, 0.58 of a pass's 1.29 ms is the fixed cost of its calls. That alone is more than twice the 0.24 ms a token takes at lesson 31's memory limit. A pass of the small model still costs 0.47 of ours, and lesson 35's draft needed under 0.35. Sixteen listeners share the calls and get 5.3 times the tokens.

## What it still cannot do

Each model in this course has read English: Shakespeare, then stories. We have not asked whether anything we built depends on that.

## Check yourself

Predict before you run: the same model, the same prompt, the top token each time, in two engines. Will the two stories be the same?

<details><summary>Answer</summary>

In our run they are, all 128 tokens. They need not be: the scores differ in about the fifth decimal place, and two top tokens that close would be a tie the engines could break differently.
</details>

<details><summary>A model has 26 times fewer weights, makes the same calls and writes 2.1 times as fast. What does that tell you?</summary>

The part that grows with the weights is about half of the time for a token. The rest is the fixed cost of each call into the engine, which a smaller model pays as often.
</details>

<details><summary>From lesson 16: why does the config file carry `rms_norm_eps` and `rope_theta`?</summary>

They are numbers inside the arithmetic that the weights file does not hold. Our code fixes them at 0.000001 and 10,000, and an engine that assumed other values would give other scores.
</details>

Go deeper:

- Apple, [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm). The engine used here, and its code for Llama-format models.
- He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). The three places a model's time goes: arithmetic, moving memory, and the cost of each call.
