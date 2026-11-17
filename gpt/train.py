"""Lesson 12. The training loop: turn one step into a run you can stop, resume and trust."""

import math
from pathlib import Path

import torch

from gpt.gpt2 import batch, evaluate, loss_on
from gpt.optim import AdamW
from gpt.results import save, say


def rate_at(step, peak, warmup, steps, floor=0.1):
    """The learning rate for one step: a straight climb to `peak`, then a slow curve down."""
    if step < warmup:
        return peak * (step + 1) / warmup
    done = (step - warmup) / max(1, steps - warmup)  # 0 at the peak, nearly 1 at the last step
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * done)))


def clip(params, limit):
    """Shrink the whole gradient to length `limit` if it is longer. Returns the length it had."""
    params = [p for p in params if p.grad is not None]
    length = torch.sqrt(sum(p.grad.pow(2).sum() for p in params)).item()
    if length > limit:
        for p in params:
            p.grad *= limit / length
    return length


def train(
    model,
    optimizer,
    ids,
    steps,
    *,
    peak,
    generator,
    start=0,
    warmup=100,
    floor=0.1,
    size=32,
    parts=1,
    limit=1.0,
    half=False,
):
    """Run steps `start` to `steps`. Yields (step, loss) after each one, so the caller can watch."""
    device = ids.device.type
    for step in range(start, steps):
        optimizer.rate = rate_at(step, peak, warmup, steps, floor)
        model.zero_grad()
        total = 0.0
        for _ in range(parts):  # a big batch, built from `parts` small ones that fit in memory
            with torch.autocast(device, dtype=torch.bfloat16, enabled=half):
                loss = loss_on(model, *batch(ids, model.context, size, generator)) / parts
            loss.backward()
            total += loss.item()
        clip(model.parameters(), limit)
        optimizer.step()
        yield step + 1, total


def checkpoint(path, model, optimizer, step, generator):
    """Everything a run needs to continue as if it had never stopped."""
    # shortcut: PyTorch's own generator, which dropout draws from, is not saved. Add
    # torch.get_rng_state() before resuming a run that uses dropout.
    state = {
        "model": model.state_dict(),
        "mean": optimizer.mean,
        "square": optimizer.square,
        "steps": optimizer.steps,
        "step": step,
        "generator": generator.get_state(),
    }
    spare = Path(path).with_suffix(".tmp")
    torch.save(state, spare)
    spare.replace(path)  # swapped in whole: a stop while writing leaves the last checkpoint alone


def resume(path, model, optimizer, generator):
    """Load a checkpoint into a fresh model, optimizer and generator. Returns the step it was at."""
    state = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(state["model"])
    optimizer.mean, optimizer.square, optimizer.steps = (
        state["mean"],
        state["square"],
        state["steps"],
    )
    generator.set_state(state["generator"].cpu())  # a generator lives on the processor
    return state["step"]


if __name__ == "__main__":
    from gpt.bigram import split
    from gpt.gpt2 import GPT2, sample
    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    train_ids, val_ids = split(ids)

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = GPT2(vocab=len(vocab), context=64, width=128, heads=4, layers=4)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    curve = []
    for step, loss in train(model, optimizer, train_ids, 5000, peak=3e-3, generator=generator):
        if step % 500 == 0:
            val = evaluate(model, val_ids)
            curve.append({"step": step, "train": loss, "val": val, "rate": optimizer.rate})
            say(f"step {step}: train {loss:.3f}, unseen {val:.3f} nats, rate {optimizer.rate:.5f}")
    val = evaluate(model, val_ids)
    story = sample(model, encode("Once upon a time", vocab), 120, generator)
    story = "".join(vocab[i] for i in story)
    say(f"unseen text: {val:.3f} nats, {val / math.log(2):.3f} bits per byte")
    say(f"sample: {story!r}")
    save(
        "12_train",
        torch=torch.__version__,
        device="cpu",
        seed=0,
        curve=curve,
        val_nats=val,
        val_bits_per_byte=val / math.log(2),
        sample=story,
    )
