"""Lesson 21. The real run: train the story model, watch it, and keep what it learned."""

import json
import math
import struct
from pathlib import Path

import torch

from gpt.checks import reads_no_future, starts_with_no_opinion
from gpt.data import read, tokenizer
from gpt.generate import generate
from gpt.gpt2 import evaluate
from gpt.model import GPT, LLAMA, load_llama
from gpt.optim import AdamW
from gpt.results import save, say
from gpt.scale import BUDGET, run, shape, steps_for
from gpt.train import checkpoint, resume, train

SHAPE = shape(256, 6)  # lesson 19: the smallest size its law cannot tell from the best
RATES = (5e-4, 1e-3, 2e-3)
CHECK = 8e13  # multiplies and adds for each short run that tries a rate


def save_llama(model, path):
    """Write the weights as a Llama-format file: lesson 10's format, with lesson 16's names."""
    tensors = {}
    for name, tensor in model.state_dict().items():
        if name != "out.weight":  # the same grid as the token rows, so stored once
            for theirs, ours in LLAMA.items():
                name = name.replace(ours, theirs)
            tensors[name] = tensor.cpu().contiguous()
    header, at = {}, 0
    for name, tensor in tensors.items():
        end = at + 4 * tensor.numel()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [at, end]}
        at = end
    head = json.dumps(header).encode()
    numbers = b"".join(tensor.numpy().tobytes() for tensor in tensors.values())
    Path(path).write_bytes(struct.pack("<Q", len(head)) + head + numbers)


def load(folder):
    """The trained story model and its tokenizer, from the folder the run below saves."""
    model = GPT(**json.loads((folder / "shape.json").read_text()))
    load_llama(model, folder / "model.safetensors")
    return model.eval(), tokenizer(folder / "vocab.json")


def long_run(cache, name, shape, ids, val_ids, watch=lambda model, step: {}):
    """A run of hours, with what lessons 12 to 14 taught: pick a rate with short runs, check the
    untrained model, then train with a checkpoint every 500 steps.

    `watch` may measure the model at each checkpoint. What it returns joins the curve.
    Returns the model, the notes kept along the way, the number of steps, and the generator.
    """
    saved, log = cache / f"{name}.pt", cache / f"{name}_log.json"

    def note():
        """Write the notes beside the old ones, then swap: a stop while writing loses nothing."""
        spare = log.with_suffix(".tmp")
        spare.write_text(json.dumps(notes))
        spare.replace(log)

    if log.exists():  # a stopped run: carry on with what it had decided and measured
        notes = json.loads(log.read_text())
    else:  # before three hours, three short runs: which rate suits this size?
        tried = [[peak, run(shape, CHECK, ids, val_ids[:200_000], peak=peak)[2]] for peak in RATES]
        notes = {"tried": tried, "peak": min(tried, key=lambda pair: pair[1])[0], "curve": []}
        note()

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = GPT(**shape).to(ids.device)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    steps = steps_for(model, BUDGET)
    if saved.exists():
        start = resume(saved, model, optimizer, generator)
    else:  # lesson 13's checks, before the long run and not after
        assert starts_with_no_opinion(model, val_ids[:20_000], shape["vocab"])
        assert reads_no_future(model, val_ids[None, :64])
        start = 0

    loop = train(
        model,
        optimizer,
        ids,
        steps,
        peak=notes["peak"],
        generator=generator,
        start=start,
        warmup=steps // 50,
    )
    for step, loss in loop:
        if step % 500 == 0 or step == steps:
            unseen = evaluate(model, val_ids[:100_000])
            point = {"step": step, "train": loss, "val": unseen} | watch(model, step)
            notes["curve"].append(point)
            print(f"step {step}: train {loss:.3f}, unseen {unseen:.3f} nats", flush=True)
            checkpoint(saved, model, optimizer, step, generator)  # a stopped run restarts here
            note()
    return model, notes, steps, generator


if __name__ == "__main__":
    cache = Path("data/cache")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = read(cache / "stories_train.bin").to(device)
    val_ids = read(cache / "stories_val.bin").to(device)
    model, notes, steps, generator = long_run(cache, "story", SHAPE, ids, val_ids)

    sizes = json.loads(Path("results/18_data.json").read_text())["sizes"]["val"]
    nats = evaluate(model, val_ids)
    bits = nats * sizes["tokens"] / sizes["bytes"] / math.log(2)
    count = sum(p.numel() for p in model.parameters())
    for peak, loss in notes["tried"]:
        say(f"{steps_for(model, CHECK)} steps at a rate of {peak:g}: {loss:.3f} nats")
    say(f"parameters: {count:,}")
    # A step reads train's batch of 32 windows.
    say(f"steps: {steps:,}, tokens: {steps * 32 * SHAPE['context']:,}, rate {notes['peak']:g}")
    for point in notes["curve"][:: max(1, len(notes["curve"]) // 9)]:
        say(f"step {point['step']}: train {point['train']:.3f}, unseen {point['val']:.3f} nats")
    say(f"unseen stories: {nats:.3f} nats a token, {bits:.3f} bits per byte")

    words = tokenizer(cache / "stories_vocab.json")
    stories = []
    for _ in range(3):
        story = generate(
            model,
            words.encode("Once upon a time"),
            250,
            stop=words.end,
            temperature=0.8,
            top_p=0.95,
            generator=generator,
        )
        stories.append(words.decode([i for i in story if i != words.end]))
        say(f"sample: {stories[-1]!r}")

    out = cache / "story"
    out.mkdir(exist_ok=True)
    save_llama(model, out / "model.safetensors")
    (out / "vocab.json").write_text((cache / "stories_vocab.json").read_text())
    (out / "shape.json").write_text(json.dumps(SHAPE))
    save(
        "21_story",
        torch=torch.__version__,
        device=device,
        seed=0,
        shape=SHAPE,
        steps=steps,
        rates_tried=notes["tried"],
        peak=notes["peak"],
        parameters=count,
        curve=notes["curve"],
        val_nats=nats,
        val_bits_per_byte=bits,
        sample=stories[0],
        samples=stories,
    )
