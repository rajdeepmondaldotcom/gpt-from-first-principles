"""Lesson 21. The real run: train the story model, watch it, and keep what it learned."""

import json
import math
import struct
from pathlib import Path

import torch

from gpt.data import read, tokenizer
from gpt.generate import generate
from gpt.gpt2 import evaluate
from gpt.model import GPT, LLAMA
from gpt.optim import AdamW
from gpt.results import save, say
from gpt.scale import shape
from gpt.train import checkpoint, resume, train

SHAPE = shape(256, 6)  # the size lesson 19 picked for this budget
STEPS, PEAK = 9000, 1e-3


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


if __name__ == "__main__":
    cache = Path("data/cache")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = read(cache / "stories_train.bin").to(device)
    val_ids = read(cache / "stories_val.bin").to(device)
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = GPT(**SHAPE).to(device)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    kept, log = cache / "story.pt", cache / "story_curve.json"
    start = resume(kept, model, optimizer, generator) if kept.exists() else 0
    curve = json.loads(log.read_text()) if start else []

    loop = train(model, optimizer, ids, STEPS, peak=PEAK, generator=generator, start=start)
    for step, loss in loop:
        if step % 500 == 0 or step == STEPS:
            unseen = evaluate(model, val_ids[:100_000])
            curve.append({"step": step, "train": loss, "val": unseen})
            print(f"step {step}: train {loss:.3f}, unseen {unseen:.3f} nats", flush=True)
            checkpoint(kept, model, optimizer, step, generator)  # a stopped run restarts from here
            log.write_text(json.dumps(curve))

    sizes = json.loads(Path("results/18_data.json").read_text())["sizes"]["val"]
    nats = evaluate(model, val_ids)
    bits = nats * sizes["tokens"] / sizes["bytes"] / math.log(2)
    count = sum(p.numel() for p in model.parameters())
    say(f"parameters: {count:,}")
    say(f"tokens trained on: {STEPS * 32 * SHAPE['context']:,}")
    for point in curve[:: max(1, len(curve) // 9)]:
        say(f"step {point['step']}: train {point['train']:.3f}, unseen {point['val']:.3f} nats")
    say(f"unseen stories: {nats:.3f} nats per token, {bits:.3f} bits per byte")

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
        steps=STEPS,
        peak=PEAK,
        parameters=count,
        curve=curve,
        val_nats=nats,
        val_bits_per_byte=bits,
        sample=stories[0],
        samples=stories,
    )
