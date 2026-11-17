"""Lesson 37. A real engine: hand our model to Apple's MLX and check that it computes the same."""

import json
import shutil

from gpt.gpt2 import read_safetensors


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
        "intermediate_size": wide.size(0),  # the width inside SwiGLU
        "max_position_embeddings": shape["context"],
        "rms_norm_eps": 1e-6,  # the three numbers lesson 16 fixed
        "rope_theta": 10000,
        "tie_word_embeddings": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def mlx_scores(folder, ids):
    """The scores Apple's engine gives a batch of token ids, as a PyTorch tensor."""
    import mlx.core as mx
    import torch
    from mlx_lm.utils import load_model

    return torch.tensor(load_model(folder)[0](mx.array(ids)).tolist())


if __name__ == "__main__":
    import time
    from pathlib import Path

    import mlx.core as mx
    import torch
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from gpt.data import read
    from gpt.engine import Engine, Request
    from gpt.generate import generate
    from gpt.model import GPT
    from gpt.results import save, say
    from gpt.scale import shape
    from gpt.story import load

    cache = Path("data/cache")
    ours, words = load(cache / "story")
    export(cache / "story", cache / "story_mlx")
    text = read(cache / "stories_val.bin")
    ids = text[:256].tolist()
    mine, theirs = ours(torch.tensor([ids]))[0], mlx_scores(cache / "story_mlx", [ids])[0]
    same = int((mine.argmax(-1) == theirs.argmax(-1)).sum())
    say(
        f"256 unseen tokens, largest gap between the two engines' scores: {(mine - theirs).abs().max():.6f}"
    )
    say(f"places where both engines put the same token on top: {same} of 256")

    prompts = [text[300 * i : 300 * i + 64].tolist() for i in range(16)]

    # How many times does one token's pass call into PyTorch? Each call has a cost of its own.
    kept = [{} for _ in ours.blocks]
    with torch.no_grad():
        ours(torch.tensor([prompts[0]]), kept)
        with torch.profiler.profile() as watched:
            ours(torch.tensor([[prompts[1][0]]]), kept)
    calls = [e for e in watched.events() if e.name.startswith("aten::")]
    outer = sum(not (e.cpu_parent and e.cpu_parent.name.startswith("aten::")) for e in calls)
    say(f"one token's pass makes {outer} calls into PyTorch")

    def fastest(work, tries=3):
        """Tokens a second for `work`, which writes and returns a count. The best of three."""
        best = 0.0
        for _ in range(tries):
            began = time.perf_counter()
            count = work()
            best = max(best, count / (time.perf_counter() - began))
        return best

    def ours_alone(device, net=ours):
        net.to(device)
        speed = fastest(lambda: len(generate(net, prompts[0], 128, temperature=0)) - 64)
        net.to("cpu")
        return speed

    def ours_batch():
        engine = Engine(ours, float("inf"), chunk=256)
        for prompt in prompts:
            engine.admit(Request(prompt, 128, how={"temperature": 0}))
        count = 0
        while engine.reading or engine.rows:
            count += len(engine.step())
        return count

    def theirs_write(model, rows):
        """Apple's engine writes 128 tokens after each prompt in `rows`, top token each time."""
        kept = make_prompt_cache(model)
        token = mx.argmax(model(mx.array(rows), cache=kept)[:, -1], axis=-1)
        written = [token.tolist()]
        for _ in range(127):
            token = mx.argmax(model(token[:, None], cache=kept)[:, -1], axis=-1)
            written.append(token.tolist())  # asking for the numbers makes the engine finish them
        return written

    numbers = {
        "ours, processor": ours_alone("cpu"),
        "ours, processor, 16 at once": fastest(ours_batch),
    }
    small = GPT(**shape(32, 2))  # untrained, with 1 weight for each 35 of the story model's
    numbers["ours, processor, a model 35 times smaller"] = ours_alone("cpu", small)
    if torch.backends.mps.is_available():
        numbers["ours, graphics chip"] = ours_alone("mps")
    for device, name in ((mx.gpu, "graphics chip"), (mx.cpu, "processor")):
        mx.set_default_device(device)
        model = load_model(cache / "story_mlx")[0]
        theirs_write(model, prompts[:1])  # a first call, to warm up
        one = fastest(lambda model=model: len(theirs_write(model, prompts[:1])))
        many = fastest(lambda model=model: 16 * len(theirs_write(model, prompts)))
        numbers[f"Apple's engine, {name}"] = one
        numbers[f"Apple's engine, {name}, 16 at once"] = many
        if device == mx.gpu:
            story = [row[0] for row in theirs_write(model, prompts[:1])]
    mine = generate(ours, prompts[0], 128, temperature=0)[64:]
    agree = next((i for i, (a, b) in enumerate(zip(mine, story)) if a != b), 128)
    for name, speed in numbers.items():
        say(f"{name}: {speed:,.0f} tokens a second")
    say(f"the two engines write the same first {agree} tokens of 128")
    save(
        "37_export",
        torch=torch.__version__,
        mlx=mx.__version__,
        tokens_a_second=numbers,
        calls=outer,
        same=agree,
    )
