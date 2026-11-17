"""Lesson 24. Evaluation: score each token once, check for leaks, and say how sure the score is."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt.data import near_repeats
from gpt.results import save, say


@torch.no_grad()
def losses(model, ids, context, stride):
    """The loss of each token after the first. Each is scored once, with up to `context` before it."""
    assert stride <= context, "a stride past the context would skip tokens"
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


def paired_gap(after, before, generator):
    """How much higher `after` is than `before`, value by value, with its 95% interval."""
    gaps = [[a - b for a, b in zip(new, old)] for new, old in zip(after, before)]
    return sum(map(sum, gaps)) / sum(map(len, gaps)), interval(gaps, generator)


def rate_interval(passed):
    """A 95% interval for a pass rate, from one True or False for each try.

    The low end is the rate at which this many passes or more still come 1 time in 40, and the high
    end the rate at which this many or fewer do. The bootstrap is too narrow when passes are few.
    """
    tries, passes = len(passed), sum(passed)

    def chance(rate, counts):
        """The chance that the number of passes is one of `counts`, at this pass rate."""
        return sum(
            math.exp(
                math.log(math.comb(tries, k))
                + k * math.log(rate)
                + (tries - k) * math.log(1 - rate)
            )
            for k in counts
        )

    def edge(counts, rising):
        """The rate at which `chance` is 1 in 40, found by halving the range 50 times."""
        low, high = 0.0, 1.0
        for _ in range(50):
            rate = (low + high) / 2
            if (chance(rate, counts) < 0.025) == rising:
                low = rate
            else:
                high = rate
        return (low + high) / 2

    return (
        edge(range(passes, tries + 1), rising=True) if passes else 0.0,
        edge(range(passes + 1), rising=False) if passes < tries else 1.0,
    )


def pass_at_k(tries, passed, k):
    """The chance that at least one of k tries passes, from `tries` tries of which `passed` did."""
    return 1 - math.comb(tries - passed, k) / math.comb(tries, k)


if __name__ == "__main__":
    from gpt.data import documents, read
    from gpt.story import load
    from gpt.tasks import attempt, count_documents, make, opening, passes, prompt

    cache = Path("data/cache")
    model, words = load(cache / "story")  # on the processor, where a run repeats
    generator = torch.Generator().manual_seed(0)

    unseen = documents(read(cache / "stories_val.bin"), words.end)
    seen = documents(read(cache / "stories_train.bin"), words.end)
    say(f"unseen stories: {len(unseen):,}, training stories: {len(seen):,}")

    # One story at a time, so the losses can be drawn story by story.
    blocks = [losses(model, doc, 256, 256).tolist() for doc in unseen]
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

    # The first 100 stories again, as one run of text: the same tokens, with more before each.
    run = torch.cat([unseen[0][:1], *(doc[1:] for doc in unseen[:100])])
    alone = sum(map(sum, blocks[:100])) / (len(run) - 1)
    say(f"100 stories, {len(run) - 1:,} tokens, each story read alone: {alone:.3f} nats a token")
    by_stride = {s: losses(model, run, 256, s).mean().item() for s in (256, 64, 16)}
    for stride, nats in by_stride.items():
        say(f"the same tokens as one run of text, stride {stride}: {nats:.3f} nats a token")

    # The same model with lesson 12's 16-bit switch on: is it worse?
    with torch.autocast("cpu", dtype=torch.bfloat16):
        halved = [losses(model, doc, 256, 256).tolist() for doc in unseen[:500]]
    gap, (low16, high16) = paired_gap(halved, blocks, generator)
    say(
        f"500 stories, 16 bits minus 32 bits: {gap:+.4f} nats, "
        f"95% interval {low16:+.4f} to {high16:+.4f}"
    )

    # A planted leak, then the real split.
    texts = [words.decode(doc[1:-1].tolist()) for doc in unseen]
    trained = [words.decode(doc[1:-1].tolist()) for doc in seen]
    planted = [*trained[:20_000], texts[3], texts[40].upper(), texts[77] + " The end."]
    say(f"three unseen stories planted among 20,000 training ones: found {leaks(planted, texts)}")
    found = len(leaks(trained, texts))
    say(f"the real split, each unseen story against each training story: {found} leaks")

    # Two tasks with an exact check, made from the unseen stories.
    counts = count_documents(trained)
    tasks = [task for task in (make(text, counts, len(trained)) for text in texts) if task]
    how = {"temperature": 0.8, "generator": generator}

    def carries_on(task):
        """The easy task: a story that opens with the name uses it again and reaches its end."""
        story, _, ended = attempt(model, words, task, lines=opening, **how)
        return ended and passes(task | {"words": []}, story)

    def to_order(task):
        """The hard task: the name and three words, asked for in three lines."""
        return attempt(model, words, task, **how)[1]

    scores = {}
    for name, passed in [("a name to carry on", carries_on), ("a story to order", to_order)]:
        first = [passed(task) for task in tasks[:100]]
        low_rate, high_rate = rate_interval(first)
        say(f"{name}: {sum(first)} of {len(first)}, 95% interval {low_rate:.2f} to {high_rate:.2f}")
        tries = [sum(passed(task) for _ in range(10)) for task in tasks[:20]]
        at = {k: sum(pass_at_k(10, n, k) for n in tries) / len(tries) for k in (1, 5)}
        say(f"{name}: pass at 1 {at[1]:.2f}, pass at 5 {at[5]:.2f}, from 10 tries on each of 20")
        scores[name] = {"passed": sum(first), "interval": [low_rate, high_rate], "tries": tries}
    say(f"a name to carry on: {opening(tasks[0])!r}")
    say(f"a story to order: {prompt(tasks[0])!r}")
    save(
        "24_evaluate",
        torch=torch.__version__,
        device="cpu",
        by_stride={str(k): v for k, v in by_stride.items()},
        alone=alone,
        val_nats=mean,
        interval=[low, high],
        sixteen_bits={"gap": gap, "interval": [low16, high16]},
        tasks=scores,
    )
