"""Lesson 27. Learning from a score: no right answer to copy, only a number for each try."""

import torch

from gpt.finetune import collate
from gpt.generate import generate
from gpt.tasks import prompt, score
from gpt.tensors import cross_entropy


def sample(model, words, task, generator):
    """One story the model writes for a task, drawn by its own chances. Returns it as a training
    example, with its score."""
    asked = words.encode(prompt(task))
    ids = generate(model, asked, 250, stop=words.end, generator=generator)
    story = words.decode([i for i in ids[len(asked) :] if i != words.end])
    return (ids, len(asked)), score(task, story)


def answer_log_chances(model, examples, pad):
    """For each example, the log of the chance the model gives its whole answer."""
    x, y, scored = collate(examples, pad, model.embed.weight.device)
    scores = model(x)
    each = -torch.stack(
        [cross_entropy(scores[row][scored[row]], y[row][scored[row]]) for row in range(len(x))]
    )
    return each * scored.sum(dim=1)  # cross_entropy gave a mean; the answer's log chance is a sum


def score_loss(log_chances, rewards, groups):
    """Raise the chance of answers that scored above their group's average, and lower the rest."""
    rewards = torch.as_tensor(rewards, dtype=log_chances.dtype, device=log_chances.device)
    groups = torch.as_tensor(groups, device=log_chances.device)
    baseline = torch.stack([rewards[groups == g].mean() for g in groups])
    return -((rewards - baseline) * log_chances).mean()


if __name__ == "__main__":
    import copy
    from pathlib import Path

    from gpt.evaluate import interval
    from gpt.finetune import measure, prepare, tune
    from gpt.optim import AdamW
    from gpt.results import save, say
    from gpt.story import load, save_llama
    from gpt.train import clip

    cache = Path("data/cache")
    model, words = load(cache / "storyteller")  # lesson 25's model
    generator = torch.Generator().manual_seed(0)
    pairs, held, unseen = prepare(cache, words)
    tasks = [task for task, _ in pairs]
    rounds, groups, tries = 100, 4, 4  # each way of learning gets 1,600 stories

    def draw(net, at):
        """Round `at`: `tries` stories for each of `groups` tasks. Examples, scores, group numbers."""
        examples, scores, group = [], [], []
        for g in range(groups):
            for _ in range(tries):
                example, mark = sample(net, words, tasks[at * groups + g], generator)
                examples.append(example)
                scores.append(mark)
                group.append(g)
        return examples, scores, group

    start, _ = measure("lesson 25's model", model, words, held, unseen, generator)

    # First try: write the stories, keep the ones that pass, and fine-tune on them as lesson 25 did.
    first, drawn = copy.deepcopy(model), []
    for at in range(rounds):
        examples, scores, _ = draw(model, at)
        drawn += [
            (tasks[at * groups + i // tries], *pair) for i, pair in enumerate(zip(examples, scores))
        ]
    kept = [example for _, example, mark in drawn if mark == 1]
    say(f"first try: {len(kept)} of {len(drawn)} stories pass, and we keep those")
    torch.save(drawn, cache / "drawn.pt")  # lessons 28 and 29 learn from these stories too
    optimizer = AdamW(first.parameters(), rate=0.0, decay=0.0)
    for _ in tune(
        first, optimizer, kept, rounds, peak=1e-4, generator=generator, pad=words.end, size=16
    ):
        pass
    keeping, _ = measure("keep what passes", first, words, held, unseen, generator)

    # Then: after each round, raise the chance of each story by how far its score is above the
    # average of its group.
    optimizer, recent = AdamW(model.parameters(), rate=1e-4, decay=0.0), []
    for at in range(rounds):
        examples, scores, group = draw(model, at)
        loss = score_loss(answer_log_chances(model, examples, words.end), scores, group)
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        recent += scores
        if (at + 1) % 20 == 0:
            say(f"round {at + 1}: the last 320 stories scored {sum(recent) / 320:.2f} on average")
            recent = []
    scoring, _ = measure("learn from the score", model, words, held, unseen, generator)

    gaps = [
        [int(a) - int(b) for a, b in zip(ours, theirs)]
        for ours, theirs in zip(scoring["by_task"], keeping["by_task"])
    ]
    gap = sum(map(sum, gaps)) / sum(map(len, gaps))
    low, high = interval(gaps, generator)
    say(
        f"asked words used, learning from the score minus keeping what passes: {gap:+.2f}, "
        f"95% interval {low:+.2f} to {high:+.2f}"
    )
    out = cache / "scored"
    out.mkdir(exist_ok=True)
    save_llama(model, out / "model.safetensors")
    for name in ("vocab.json", "shape.json"):
        (out / name).write_text((cache / "story" / name).read_text())
    save(
        "27_reinforce",
        torch=torch.__version__,
        device="cpu",
        start=start,
        keeping=keeping,
        scoring=scoring,
        kept=len(kept),
        gap=gap,
        interval=[low, high],
    )
