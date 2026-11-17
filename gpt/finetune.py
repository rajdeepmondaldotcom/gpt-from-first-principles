"""Lesson 25. Fine-tuning: teach a trained model to answer a prompt, and to stop."""

from pathlib import Path

import torch

from gpt.data import documents, read
from gpt.evaluate import interval, losses, rate_interval
from gpt.results import save, say
from gpt.story import save_llama
from gpt.tasks import attempt, count_documents, make, prompt, used
from gpt.tensors import cross_entropy
from gpt.train import clip, rate_at

STEPS = 2000  # steps of 32 examples in lessons 25 and 26
WARMUP = 20  # steps over which each fine-tune's rate climbs to its peak


def example(words, ask, answer):
    """One training example: the prompt, the answer, the end token. Returns ids and the answer's start."""
    asked = words.encode(ask)
    return [*asked, *words.encode(answer), words.end], len(asked)


def collate(examples, pad, device="cpu", width=None):
    """Stack examples of different lengths. `scored` marks the places whose target is answer."""
    width = width or max(len(ids) for ids, _ in examples) - 1  # as wide as the longest needs
    x = torch.full((len(examples), width), pad, device=device)
    y, scored = x.clone(), torch.zeros_like(x, dtype=torch.bool)
    for row, (ids, start) in enumerate(examples):
        ids = torch.tensor(ids, device=device)
        x[row, : len(ids) - 1], y[row, : len(ids) - 1] = ids[:-1], ids[1:]
        # place start - 1 predicts the answer's first token
        scored[row, start - 1 : len(ids) - 1] = True
    return x, y, scored


def answer_loss(model, x, y, scored):
    """Lesson 10's loss, counted only where the target is part of an answer."""
    return cross_entropy(model(x)[scored], y[scored])


def tune(model, optimizer, examples, steps, *, peak, generator, pad, size=16):
    """Lesson 12's loop on batches of examples. Yields (step, loss)."""
    device = model.embed.weight.device
    for step in range(steps):
        picks = torch.randint(len(examples), (size,), generator=generator).tolist()
        optimizer.rate = rate_at(step, peak, WARMUP, steps)
        # Each batch is padded to one width: the graphics chip stalled when the shape kept changing.
        chosen = [examples[i] for i in picks]
        loss = answer_loss(model, *collate(chosen, pad, device, width=model.context - 1))
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        yield step + 1, loss.item()


def prepare(cache, words):
    """Training stories with the task each one answers, 200 tasks from unseen stories, and 500
    unseen stories as ids."""
    seen = documents(read(cache / "stories_train.bin")[:20_000_000], words.end)
    unseen = documents(read(cache / "stories_val.bin"), words.end)
    seen_text = [words.decode(doc[1:-1].tolist()) for doc in seen]
    counts = count_documents(seen_text)
    pairs = [(make(story, counts, len(seen_text)), story) for story in seen_text]
    unseen_text = (words.decode(doc[1:-1].tolist()) for doc in unseen)
    held = [task for task in (make(story, counts, len(seen_text)) for story in unseen_text) if task]
    return [pair for pair in pairs if pair[0]], held[:200], unseen[:500]


def examples_from(pairs, words, context):
    """Each pair as a training example: the task's prompt, then its story. Only those that fit."""
    made = (example(words, prompt(task), " " + story) for task, story in pairs)
    return [one for one in made if len(one[0]) <= context]


def measure(name, model, words, held, unseen, generator):
    """Tasks passed, asked words used, stories that end by themselves, and the loss on plain
    unseen stories. Returns the numbers, and each plain story's losses."""
    tries = [attempt(model, words, task, temperature=0.8, generator=generator) for task in held]
    passed, ended = [ok for _, ok, _ in tries], sum(end for _, _, end in tries)
    low, high = rate_interval(passed)
    words_used = [used(task, story) for task, (story, _, _) in zip(held, tries)]
    share = sum(map(sum, words_used)) / sum(map(len, words_used))
    low_share, high_share = interval(words_used, generator)  # drawn task by task
    plain = [losses(model, doc, 256, 256).tolist() for doc in unseen]
    nats = sum(map(sum, plain)) / sum(map(len, plain))
    say(f"{name}: {sum(passed)} of {len(held)} tasks passed, 95% interval {low:.2f} to {high:.2f}")
    say(
        f"{name}: {share:.2f} of the asked words used, 95% interval {low_share:.2f} to {high_share:.2f}"
    )
    say(f"{name}: {ended} stories end by themselves, plain stories {nats:.3f} nats a token")
    numbers = {"passed": sum(passed), "words_used": share, "ended": ended, "plain_nats": nats}
    return numbers | {"by_task": words_used}, plain


def keep(model, cache, name):
    """Save a tuned model as a folder `load` reads, with the story model's tokenizer and sizes."""
    out = cache / name
    out.mkdir(exist_ok=True)
    save_llama(model, out / "model.safetensors")
    for file in ("vocab.json", "shape.json"):
        (out / file).write_text((cache / "story" / file).read_text())


def paired_gap(after, before, generator):
    """How much higher `after` is than `before`, value by value, with its 95% interval."""
    gaps = [[a - b for a, b in zip(new, old)] for new, old in zip(after, before)]
    return sum(map(sum, gaps)) / sum(map(len, gaps)), interval(gaps, generator)


if __name__ == "__main__":
    from gpt.optim import AdamW
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "story")
    generator = torch.Generator().manual_seed(0)
    pairs, held, unseen = prepare(cache, words)
    examples = examples_from(pairs, words, model.context)
    say(f"examples to learn from: {len(examples):,}")

    before, plain_before = measure("before", model, words, held, unseen, generator)
    # Train on the graphics chip. Writing and scoring stay on the processor, where they are faster.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    for step, loss in tune(
        model, optimizer, examples, STEPS, peak=1e-3, generator=generator, pad=words.end, size=32
    ):
        if step % 250 == 0:
            say(f"step {step}: {loss:.3f} nats on the answers")
    model.to("cpu")
    after, plain_after = measure("after", model, words, held, unseen, generator)
    gap, (low, high) = paired_gap(plain_after, plain_before, generator)
    say(
        f"plain stories, after minus before: {gap:+.3f} nats, 95% interval {low:+.3f} to {high:+.3f}"
    )
    task = {"name": "Mira", "words": ["cat", "moon", "brave"]}
    story, ok, _ = attempt(model, words, task, temperature=0.8, generator=generator)
    say(f"to order, passes {ok}: {story.strip()!r}")

    keep(model, cache, "storyteller")
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
