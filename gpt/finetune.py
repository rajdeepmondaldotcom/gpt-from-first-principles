"""Lesson 25. Fine-tuning: teach a trained model to answer a prompt, and to stop."""

from pathlib import Path

import torch

from gpt.data import read
from gpt.evaluate import documents, interval, losses
from gpt.results import save, say
from gpt.tasks import attempt, count_documents, make, prompt
from gpt.tensors import cross_entropy
from gpt.train import clip, rate_at


def example(words, ask, answer):
    """One training example: the prompt, the answer, the end token. Returns ids and the answer's start."""
    prompt = words.encode(ask)
    return [*prompt, *words.encode(answer), words.end], len(prompt)


def collate(examples, pad, device="cpu"):
    """Stack examples of different lengths. `scored` marks the places whose target is answer."""
    longest = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), longest - 1), pad, device=device)
    y, scored = x.clone(), torch.zeros_like(x, dtype=torch.bool)
    for row, (ids, start) in enumerate(examples):
        ids = torch.tensor(ids, device=device)
        x[row, : len(ids) - 1], y[row, : len(ids) - 1] = ids[:-1], ids[1:]
        scored[row, start - 1 : len(ids) - 1] = (
            True  # place start - 1 predicts the answer's first token
        )
    return x, y, scored


def answer_loss(model, x, y, scored):
    """Lesson 10's loss, counted only where the target is part of an answer."""
    return cross_entropy(model(x)[scored], y[scored])


def tune(model, optimizer, examples, steps, *, peak, generator, pad, size=16, warmup=20):
    """Lesson 12's loop on batches of examples. Yields (step, loss)."""
    device = model.embed.weight.device
    for step in range(steps):
        picks = torch.randint(len(examples), (size,), generator=generator).tolist()
        optimizer.rate = rate_at(step, peak, warmup, steps)
        loss = answer_loss(model, *collate([examples[i] for i in picks], pad, device))
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        yield step + 1, loss.item()


def prepare(cache, words):
    """Examples to learn from, 100 tasks from unseen stories, and 500 unseen stories as ids."""
    seen = documents(read(cache / "stories_train.bin")[:5_000_000], words.end)
    unseen = documents(read(cache / "stories_val.bin"), words.end)
    seen_text = [words.decode(doc[1:-1].tolist()) for doc in seen]
    counts = count_documents(seen_text)
    examples = [
        example(words, prompt(task), " " + story)
        for task, story in ((make(story, counts, len(seen_text)), story) for story in seen_text)
        if task
    ]
    unseen_text = (words.decode(doc[1:-1].tolist()) for doc in unseen)
    held = [task for task in (make(story, counts, len(seen_text)) for story in unseen_text) if task]
    return examples, held[:100], unseen[:500]


def measure(name, model, words, held, unseen, generator):
    """Tasks passed, stories that end by themselves, and the loss on plain unseen stories."""
    device = model.embed.weight.device
    tries = [attempt(model, words, task, temperature=0.8, generator=generator) for task in held]
    passed, ended = [int(ok) for _, ok, _ in tries], sum(end for _, _, end in tries)
    low, high = interval([[x] for x in passed], generator)
    plain = [losses(model, doc.to(device), 256, 256).tolist() for doc in unseen]
    nats = sum(map(sum, plain)) / sum(map(len, plain))
    say(
        f"{name}: {sum(passed)} of {len(held)} tasks passed (95% interval {low:.2f} to {high:.2f}), "
        f"{ended} stories end by themselves, plain stories {nats:.3f} nats a token"
    )
    return {"passed": sum(passed), "ended": ended, "plain_nats": nats}, plain


def plain_gap(after, before, generator):
    """How much worse plain stories got, token by token, with its 95% interval."""
    gaps = [[a - b for a, b in zip(new, old)] for new, old in zip(after, before)]
    return sum(map(sum, gaps)) / sum(map(len, gaps)), interval(gaps, generator)


if __name__ == "__main__":
    from gpt.optim import AdamW
    from gpt.story import load, save_llama

    cache = Path("data/cache")
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # where it trains
    model, words = load(cache / "story")  # it writes and is measured on the processor
    generator = torch.Generator().manual_seed(0)
    examples, held, unseen = prepare(cache, words)
    say(f"examples to learn from: {len(examples):,}")

    before, plain_before = measure("before", model, words, held, unseen, generator)
    model.to(device)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    for step, loss in tune(
        model, optimizer, examples, 600, peak=3e-4, generator=generator, pad=words.end
    ):
        if step % 100 == 0:
            say(f"step {step}: {loss:.3f} nats on the answers")
    model.to("cpu")
    after, plain_after = measure("after", model, words, held, unseen, generator)
    gap, (low, high) = plain_gap(plain_after, plain_before, generator)
    say(
        f"plain stories, after minus before: {gap:+.3f} nats, 95% interval {low:+.3f} to {high:+.3f}"
    )
    task = {"name": "Mira", "words": ["cat", "moon", "brave"]}
    story, ok, _ = attempt(model, words, task, temperature=0.8, generator=generator)
    say(f"to order, passes {ok}: {story.strip()!r}")

    out = cache / "storyteller"
    out.mkdir(exist_ok=True)
    save_llama(model, out / "model.safetensors")
    for name in ("vocab.json", "shape.json"):
        (out / name).write_text((cache / "story" / name).read_text())
    save(
        "25_finetune",
        torch=torch.__version__,
        device=device,
        before=before,
        after=after,
        plain_gap=gap,
        plain_interval=[low, high],
        sample=story.strip(),
    )
