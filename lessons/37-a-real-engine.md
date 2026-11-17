# 37. A real engine

By the end of this page our model runs in Apple's MLX engine, gives the scores it gives in ours, and you have timed both engines on both of this Mac's chips.

Our server gives the answers the model gives alone, and we wrote its Python for people to read. We have not asked how fast an engine built for speed would run the same model.

## The idea

Our model is one file of weights and about 200 lines of Python. An engine is another program that reads the same file and does the same arithmetic, written by people whose whole job is speed. Handing a model over takes two things: the engine must find each weight, and it must do with them what we do.

Finding the weights is settled already. Lesson 21 saved them under the names that engines for Llama-format models look for, in lesson 10's file format. What the engine still needs is the sizes: the width, the context, the number of blocks, of heads and of key heads, and the three settings that lesson 16 said our code fixes. Those are the width inside SwiGLU, the small number inside RMSNorm and the base of the rotary speeds. Those go in a small text file called `config.json`. Writing that file is the whole export.

Doing what we do is harder to see. The engine has its own code for attention, for rotary positions, for RMSNorm and for SwiGLU. One detail done another way, such as which entries lesson 15's rotation pairs up, and its scores would differ from ours with no error shown. So we do not ask whether the file loads. We ask whether the scores are the same: the same tokens go into both engines, and we compare each score. They cannot agree to the last bit, because the two engines add the same numbers in different orders and 32-bit sums depend on the order.

Then speed, like against like. Both engines write 128 tokens after the same prompt and take the top token each time, so they write the same story. We time four corners: our engine or Apple's, on the processor or on the graphics chip. We time one reader, and sixteen at once.

One more timing helps to read the rest. In our engine one token's pass makes about 500 calls into PyTorch, and each call costs some time whatever the size of its arithmetic. If that fixed cost is most of a pass, a far smaller model will not be much faster. So the script also times a model with 35 times fewer weights.

## By hand

The story model's sizes, as the config file gives them.

| Name in the file | Value | Where it comes from |
|---|---|---|
| `hidden_size` | 256 | the width |
| `num_hidden_layers` | 6 | the blocks |
| `num_attention_heads` | 8 | a head for each 32 of width |
| `num_key_value_heads` | 4 | lesson 16's grouped heads |
| `max_position_embeddings` | 256 | the context |
| `intermediate_size` | 682 | SwiGLU's width: 8 × 256 / 3, rounded down |
| `vocab_size` | 4,097 | 4,096 tokens and the end token |

From tokens a second to the time for one token. At @@OURS@@ tokens a second, one token takes 1,000 / @@OURS@@ = @@MS@@ ms. The story model has 6 blocks, so that is about @@PERBLOCK@@ ms a block.

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

`shape.json` is the file lesson 21 saved beside the weights. `read_safetensors` is lesson 10's reader. `gate_proj` is the Llama name for SwiGLU's first grid, and a grid is stored with a row for each output (lesson 4), so its count of rows is SwiGLU's width. A `kv_heads` of None means a key head for each head. `tie_word_embeddings` says that the token rows and the scores layer are one grid, as they have been since lesson 10.

## The proof

```bash
uv run pytest tests/test_export.py
```

- The config file of a small model names each of its sizes, and the weights file is copied byte for byte.

One more test needs Apple's engine, which exists for Macs. As in lesson 5, `--extra reference` installs it and `-m reference` picks the tests that use it:

```bash
uv run --extra reference pytest -m reference tests/test_export.py
```

- A small model with random weights and grouped heads, loaded in MLX, gives our scores to 0.0001, and the same top token at each place.

```bash
uv run python -m gpt.export
```

```
@@OUTPUT@@
```

Timings are from an M4 laptop that was running other jobs, and they move from run to run.

@@RESULTS@@

## What it still cannot do

Each model in this course has read English: Shakespeare, then stories. We have not asked whether anything we built depends on that.

## Check yourself

Predict before you run: the same model, the same prompt, the top token each time, in two engines. Will the two stories be the same?

<details><summary>Answer</summary>

They can differ, and in our run they did not. Scores differ in about the fifth decimal place. Two top tokens that close would be a tie that the engines could break differently, and from there the stories would part.
</details>

<details><summary>An engine loads our file with no error, and its stories read well. Why is that not proof that the export is right?</summary>

A model with one detail done differently still writes plausible text. Only a comparison of scores, token by token, shows that the arithmetic is ours.
</details>

<details><summary>A model has 35 times fewer weights and writes only a few times faster. What does that tell you?</summary>

Most of the time for a token is not arithmetic on weights. It is the fixed cost of each call into the engine, which a smaller model pays as often.
</details>

<details><summary>From lesson 16: why does the config file carry `rms_norm_eps` and `rope_theta`?</summary>

They are numbers inside the arithmetic that the weights file does not hold. Our code fixes them at 0.000001 and 10,000, and an engine that assumed other values would give other scores.
</details>

Go deeper:

- Apple, [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm). The engine used here, and its code for Llama-format models.
- He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). The three places a model's time goes: arithmetic, moving memory, and the cost of each call.
- [llama.cpp](https://github.com/ggml-org/llama.cpp). Another engine, with a file format of its own, that runs the same kind of model on a processor.
