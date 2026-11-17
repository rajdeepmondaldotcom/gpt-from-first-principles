"""Lesson 29. Preferences without a scorer: learn straight from which of two answers is better."""

import torch
from torch.nn.functional import logsigmoid


def preference_loss(ours_better, ours_worse, frozen_better, frozen_worse, beta=0.1):
    """DPO. How much more we favour the better answer than a frozen copy of the model does."""
    lead = (ours_better - frozen_better) - (ours_worse - frozen_worse)
    return -logsigmoid(beta * lead).mean()


if __name__ == "__main__":
    from pathlib import Path

    from gpt.evaluate import interval
    from gpt.finetune import keep, measure, paired_gap, prepare
    from gpt.optim import AdamW
    from gpt.reinforce import answer_log_chances
    from gpt.results import save, say
    from gpt.scorer import liked, split
    from gpt.story import load
    from gpt.train import clip

    cache = Path("data/cache")
    model, words = load(cache / "storyteller")  # lesson 25's model
    generator = torch.Generator().manual_seed(0)
    drawn = torch.load(cache / "drawn.pt")  # lesson 27's stories: 4 for each order
    learn, held, _ = split(drawn, words)
    flat = [pair for pairs in held for pair in pairs]
    tokens = [len(ids) - start - (ids[-1] == words.end) for _, (ids, start), _ in drawn]
    say(
        f"pairs to learn from: {len(learn):,}, pairs held back: {len(flat)}. "
        f"A story is {sum(tokens) / len(tokens):.0f} tokens on average"
    )
    _, tasks, unseen = prepare(cache, words)

    def log_chances(stories):
        """The log chance the model gives each story now, eight stories to a pass."""
        return torch.cat(
            [
                answer_log_chances(model, stories[at : at + 8], words.end)
                for at in range(0, len(stories), 8)
            ]
        )

    def sides(pairs):
        """The log chances of each pair's better story, and of each pair's worse one."""
        return [log_chances([pair[side] for pair in pairs]) for side in (0, 1)]

    with torch.no_grad():  # the reference is the model as it is now: its side is worked out once
        fixed, fixed_held = sides(learn), sides(flat)

    start, plain_start = measure("lesson 25's model", model, words, tasks, unseen, generator)
    liked_start = liked(model, words, tasks, generator)
    optimizer = AdamW(model.parameters(), rate=1e-5, decay=0.0)
    for step in range(200):
        picks = torch.randint(len(learn), (8,), generator=generator)
        better, worse = zip(*(learn[i] for i in picks.tolist()))
        ours = [answer_log_chances(model, list(part), words.end) for part in (better, worse)]
        loss = preference_loss(ours[0], ours[1], fixed[0][picks], fixed[1][picks])
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            say(f"step {step + 1}: {loss.item():.3f} on 8 pairs. ln 2 is 0.693")
    with torch.no_grad():
        ours = sides(flat)
    lead = ((ours[0] - fixed_held[0]) - (ours[1] - fixed_held[1])).tolist()
    gained = [[lead.pop(0) > 0 for _ in pairs] for pairs in held]  # order by order
    low, high = interval([pairs for pairs in gained if pairs], generator)
    say(
        f"held-back pairs where the better story gained on the worse one: "
        f"{sum(map(sum, gained))} of {len(flat)}, 95% interval {low:.2f} to {high:.2f}"
    )
    after, plain_after = measure("after the pairs", model, words, tasks, unseen, generator)
    gap, (low_gap, high_gap) = paired_gap(plain_after, plain_start, generator)
    say(
        f"plain stories, after minus before: {gap:+.3f} nats, "
        f"95% interval {low_gap:+.3f} to {high_gap:+.3f}"
    )
    liked_after = liked(model, words, tasks, generator)
    low_liked, high_liked = interval([[a - b] for a, b in zip(liked_after, liked_start)], generator)
    say(
        f"the readers give its stories {sum(liked_after) / len(tasks):.2f} on average, where they "
        f"gave {sum(liked_start) / len(tasks):.2f}. The change has a 95% interval of "
        f"{low_liked:+.2f} to {high_liked:+.2f}"
    )
    keep(model, cache, "preferred")
    save(
        "29_prefer",
        torch=torch.__version__,
        device="cpu",
        pairs={"learn": len(learn), "held": len(flat)},
        gained={"count": sum(map(sum, gained)), "interval": [low, high]},
        start=start,
        after=after,
        plain_gap=gap,
        readers={"start": sum(liked_start) / len(tasks), "after": sum(liked_after) / len(tasks)},
    )
