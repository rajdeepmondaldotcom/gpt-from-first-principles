"""Lesson 24. Evaluation: score each token once, check for leaks, and say how sure the score is."""

import math
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt.data import near_repeats
from gpt.results import save, say


@torch.no_grad()
def losses(model, ids, context, stride):
    """The loss of each token after the first. Each is scored once, with up to `context` before it."""
    out = []
    for start in range(0, len(ids) - 1, stride):
        end = min(start + stride, len(ids) - 1)  # this pass scores tokens start + 1 to end
        window = ids[max(0, end - context) : end + 1]
        scores = model(window[None, :-1])[0, start - end :]  # only the places not scored before
        out.append(F.cross_entropy(scores, window[start - end :].long(), reduction="none"))
    return torch.cat(out)


def leaks(seen, unseen):
    """Indexes of unseen documents that repeat, or nearly repeat, a document from training."""
    repeats = near_repeats([*seen, *unseen])
    return sorted(i - len(seen) for i, j in repeats.items() if i >= len(seen) > j)


def interval(blocks, generator, draws=1000):
    """A 95% interval for the mean of every value. Whole blocks are drawn, never single values."""
    sums = torch.tensor([float(sum(block)) for block in blocks])
    counts = torch.tensor([float(len(block)) for block in blocks])
    drawn = torch.randint(len(blocks), (draws, len(blocks)), generator=generator)
    means = sums[drawn].sum(dim=1) / counts[drawn].sum(dim=1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def documents(ids, end):
    """Cut a run of token ids into its documents, each with the end token before and after."""
    mark = torch.tensor([end], dtype=ids.dtype)
    cuts = [-1, *(ids == end).nonzero().flatten().tolist()]
    return [torch.cat([mark, ids[a + 1 : b + 1]]) for a, b in pairwise(cuts) if b - a > 1]


def pass_at_k(tries, passed, k):
    """The chance that at least one of k tries passes, from `tries` tries of which `passed` did."""
    return 1 - math.comb(tries - passed, k) / math.comb(tries, k)


if __name__ == "__main__":
    from gpt.data import read
    from gpt.story import load
    from gpt.tasks import attempt, count_documents, make, prompt

    cache = Path("data/cache")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, words = load(cache / "story", device)
    generator = torch.Generator().manual_seed(0)

    unseen = documents(read(cache / "stories_val.bin"), words.end)
    seen = documents(read(cache / "stories_train.bin")[:5_000_000], words.end)
    say(f"unseen documents: {len(unseen):,}, and {len(seen):,} of the training ones")

    # Each token scored once, with more or less of the text before it.
    run = torch.cat(unseen)[:20_000].to(device)
    by_stride = {s: losses(model, run, 256, s).mean().item() for s in (256, 64, 16)}
    for stride, nats in by_stride.items():
        say(f"20,000 tokens at a stride of {stride}: {nats:.3f} nats a token")

    # One document at a time, so the losses can be drawn document by document.
    blocks = [losses(model, doc.to(device), 256, 256).tolist() for doc in unseen]
    mean = sum(map(sum, blocks)) / sum(map(len, blocks))
    low, high = interval(blocks, generator)
    say(f"unseen stories: {mean:.3f} nats a token, 95% interval {low:.3f} to {high:.3f}")
    part = blocks[:250]  # the same 250 stories, drawn two ways
    by_story = interval(part, generator)
    by_token = interval([[x] for block in part for x in block], generator)
    say(
        f"250 stories, drawn story by story: {by_story[1] - by_story[0]:.3f} wide. "
        f"Drawn token by token: {by_token[1] - by_token[0]:.3f} wide"
    )

    # The same model with its forward pass in 16 bits: is it worse?
    some = unseen[:500]
    with torch.autocast(device, dtype=torch.bfloat16):
        halved = [losses(model, doc.to(device), 256, 256).tolist() for doc in some]
    gaps = [[h - f for h, f in zip(half, full)] for half, full in zip(halved, blocks)]
    gap = sum(map(sum, gaps)) / sum(map(len, gaps))
    low16, high16 = interval(gaps, generator)
    say(f"16 bits minus 32 bits: {gap:+.4f} nats, 95% interval {low16:+.4f} to {high16:+.4f}")

    # A planted leak, then the real split.
    texts = [words.decode(doc[1:-1].tolist()) for doc in unseen]
    trained = [words.decode(doc[1:-1].tolist()) for doc in seen]
    planted = [*trained, texts[3], texts[40].upper(), texts[77] + " The end."]
    say(f"three unseen stories planted in the training text: found {leaks(planted, texts)}")
    say(f"the real split: {len(leaks(trained, texts))} leaks")

    # A task with an exact check: use this name and these three words.
    counts = count_documents(trained)
    tasks = [task for task in (make(text, counts, len(trained)) for text in texts) if task]

    def passed(task):
        return attempt(model, words, task, temperature=0.8, generator=generator)[1]

    first = [passed(task) for task in tasks[:100]]
    low_task, high_task = interval([[int(x)] for x in first], generator)
    say(f"tasks passed: {sum(first)} of 100, 95% interval {low_task:.2f} to {high_task:.2f}")
    tries = [sum(passed(task) for _ in range(10)) for task in tasks[:20]]
    for k in (1, 5):
        chance = sum(pass_at_k(10, passed, k) for passed in tries) / len(tries)
        say(f"pass at {k}, from 10 tries on each of 20 tasks: {chance:.2f}")
    say(f"a task: {prompt(tasks[0])!r}")
    save(
        "24_evaluate",
        torch=torch.__version__,
        device=device,
        by_stride={str(k): v for k, v in by_stride.items()},
        val_nats=mean,
        interval=[low, high],
        sixteen_bits={"gap": gap, "interval": [low16, high16]},
        tasks_passed=sum(first),
        tries=tries,
    )
