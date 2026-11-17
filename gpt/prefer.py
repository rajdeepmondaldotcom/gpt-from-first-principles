"""Lesson 29. Preferences without a scorer: learn straight from which of two answers is better."""

import torch
from torch.nn.functional import logsigmoid


def preference_loss(ours_better, ours_worse, frozen_better, frozen_worse, beta=0.1):
    """DPO. How much more we favour the better answer than a frozen copy of the model does."""
    lead = (ours_better - frozen_better) - (ours_worse - frozen_worse)
    return -logsigmoid(beta * lead).mean()


if __name__ == "__main__":
    import copy
    from pathlib import Path

    from gpt.evaluate import rate_interval
    from gpt.finetune import measure, plain_gap, prepare
    from gpt.optim import AdamW
    from gpt.reinforce import answer_log_chances
    from gpt.results import save, say
    from gpt.scorer import pairs_from
    from gpt.story import load, save_llama
    from gpt.train import clip

    cache = Path("data/cache")
    model, words = load(cache / "storyteller")  # lesson 25's model
    frozen = copy.deepcopy(model)  # the same model, never trained again: the reference
    generator = torch.Generator().manual_seed(0)
    drawn = torch.load(cache / "drawn.pt")  # lesson 27's 1,600 stories: 4 for each of 400 tasks
    learn, held = pairs_from(drawn[:1360], words), pairs_from(drawn[1360:], words)
    say(f"pairs to learn from: {len(learn):,}, pairs held back: {len(held)}")
    _, tasks, unseen = prepare(cache, words)

    def leads(net, pairs):
        """For each pair, how much more than the frozen model `net` favours the better story."""
        out = []
        for at in range(0, len(pairs), 8):
            better, worse = zip(*pairs[at : at + 8])
            ours = [answer_log_chances(net, list(part), words.end) for part in (better, worse)]
            with torch.no_grad():
                theirs = [
                    answer_log_chances(frozen, list(part), words.end) for part in (better, worse)
                ]
            out.append((ours[0] - theirs[0]) - (ours[1] - theirs[1]))
        return torch.cat(out)

    start, plain_start = measure("lesson 25's model", model, words, tasks, unseen, generator)
    optimizer = AdamW(model.parameters(), rate=1e-5, decay=0.0)
    for step in range(200):
        picks = torch.randint(len(learn), (8,), generator=generator).tolist()
        better, worse = zip(*(learn[i] for i in picks))
        with torch.no_grad():
            theirs = [answer_log_chances(frozen, list(part), words.end) for part in (better, worse)]
        ours = [answer_log_chances(model, list(part), words.end) for part in (better, worse)]
        loss = preference_loss(ours[0], ours[1], theirs[0], theirs[1])
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            say(f"step {step + 1}: {loss.item():.3f} on 8 pairs. ln 2 is 0.693")
    with torch.no_grad():
        gained = (leads(model, held) > 0).tolist()
    low, high = rate_interval(gained, generator)
    say(
        f"held-back pairs where the better story gained on the worse one: {sum(gained)} of "
        f"{len(held)}, 95% interval {low:.2f} to {high:.2f}"
    )
    after, plain_after = measure("after the pairs", model, words, tasks, unseen, generator)
    gap, (low_gap, high_gap) = plain_gap(plain_after, plain_start, generator)
    say(
        f"plain stories, after minus before: {gap:+.3f} nats, "
        f"95% interval {low_gap:+.3f} to {high_gap:+.3f}"
    )
    out = cache / "preferred"
    out.mkdir(exist_ok=True)
    save_llama(model, out / "model.safetensors")
    for name in ("vocab.json", "shape.json"):
        (out / name).write_text((cache / "story" / name).read_text())
    save(
        "29_prefer",
        torch=torch.__version__,
        device="cpu",
        pairs={"learn": len(learn), "held": len(held)},
        gained={"count": sum(gained), "interval": [low, high]},
        start=start,
        after=after,
        plain_gap=gap,
    )
