"""Lesson 28. A model of what people prefer: learn a score from pairs where one text is better."""

import re

import torch
from torch import nn

from gpt.tasks import score


class Scorer(nn.Module):
    """A GPT's body with one new layer on top. It reads a text and gives it a single number."""

    def __init__(self, body):
        super().__init__()
        self.body = body
        self.head = nn.Linear(body.embed.weight.size(1), 1)

    def forward(self, ids, last):
        """ids: (batch, tokens), padded. last: the place of each row's final real token."""
        x = self.body.embed(ids)
        for block in self.body.blocks:
            x = block(x)
        read = self.body.norm(x)[torch.arange(len(ids)), last]  # that place has read the whole text
        return self.head(read).squeeze(-1)


def taste(task, text, ended):
    """A stand-in for readers, as a number. They want a story that ends, runs to 40 words or more
    and says no sentence twice. Among those, they like the name and the asked words."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    whole = ended and len(text.split()) >= 40 and len(set(sentences)) == len(sentences)
    return score(task, text) if whole else 0.0


def readers(words, task, example):
    """What the readers give a story the model wrote, held as one of lesson 25's examples."""
    ids, start = example
    text = words.decode([i for i in ids[start:] if i != words.end])
    return taste(task, text, ids[-1] == words.end)


def pairs_from(drawn, words, tries=4):
    """From each task's `tries` stories, each pair where the readers like one story more.

    Returns (better, worse) pairs of examples. An example is lesson 25's: the token ids of
    prompt and story, and the place where the story starts.
    """
    out = []
    for g in range(0, len(drawn), tries):
        marks = [(readers(words, task, one), one) for task, one, _ in drawn[g : g + tries]]
        for i, (mine, a) in enumerate(marks):
            out += [(a, b) for theirs, b in marks[i + 1 :] if mine > theirs]
            out += [(b, a) for theirs, b in marks[i + 1 :] if mine < theirs]
    return out


def pad(texts, fill, device="cpu"):
    """Stack lists of token ids of different lengths. Returns the grid and each row's last place."""
    ids = torch.full((len(texts), max(map(len, texts))), fill, device=device)
    for row, text in enumerate(texts):
        ids[row, : len(text)] = torch.tensor(text, device=device)
    return ids, torch.tensor([len(text) - 1 for text in texts], device=device)


def comparison_loss(better, worse):
    """Low when the better text of each pair scores higher. ln 2 when the scorer cannot tell."""
    return -nn.functional.logsigmoid(better - worse).mean()


if __name__ == "__main__":
    import copy
    from pathlib import Path

    from gpt.evaluate import rate_interval
    from gpt.finetune import prepare
    from gpt.optim import AdamW
    from gpt.reinforce import answer_log_chances, sample, score_loss
    from gpt.results import save, say
    from gpt.story import load
    from gpt.train import clip

    cache = Path("data/cache")
    model, words = load(cache / "storyteller")  # lesson 25's model
    generator = torch.Generator().manual_seed(0)
    drawn = torch.load(cache / "drawn.pt")  # lesson 27's 1,600 stories: 4 for each of 400 tasks
    learn, held = pairs_from(drawn[:1360], words), pairs_from(drawn[1360:], words)
    say(
        f"pairs to learn from: {len(learn):,}, from 340 orders. Pairs held back: {len(held)}, from 60"
    )

    scorer = Scorer(copy.deepcopy(model))  # it starts from what the story model knows

    def scores(examples):
        return scorer(*pad([ids for ids, _ in examples], words.end))

    @torch.no_grad()
    def right(pairs):
        """For each pair, whether the scorer gives the better story the higher number."""
        return [bool(scores([a]) > scores([b])) for a, b in pairs]

    before = sum(right(held))
    optimizer = AdamW(scorer.parameters(), rate=1e-4, decay=0.0)
    for step in range(300):
        picks = torch.randint(len(learn), (16,), generator=generator).tolist()
        loss = comparison_loss(
            scores([learn[i][0] for i in picks]), scores([learn[i][1] for i in picks])
        )
        scorer.zero_grad()
        loss.backward()
        clip(scorer.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            say(f"step {step + 1}: {loss.item():.3f} on 16 pairs. ln 2 is 0.693")
    ranked = right(held)
    low, high = rate_interval(ranked, generator)
    say(f"held-back pairs ranked right before any training: {before} of {len(held)}")
    say(f"after: {sum(ranked)} of {len(held)}, 95% interval {low:.2f} to {high:.2f}")

    # Now push lesson 25's model against the scorer. Its number is the reward.
    with torch.no_grad():
        base = {
            "scorer": sum(scores([one]).item() for _, one, _ in drawn[1360:]) / 240,
            "readers": sum(readers(words, task, one) for task, one, _ in drawn[1360:]) / 240,
        }
    say(
        f"before the push, the 240 stories held back: the scorer gives them "
        f"{base['scorer']:+.2f} on average, the readers {base['readers']:.2f}"
    )
    tasks = [task for task, _ in prepare(cache, words)[0]][400:]  # tasks lesson 27 did not use
    optimizer, rounds, curve = AdamW(model.parameters(), rate=3e-4, decay=0.0), 100, []
    learned, liked = [], []
    for at in range(rounds):
        examples, group, picked = [], [], []
        for g in range(4):
            for _ in range(4):
                examples.append(sample(model, words, tasks[4 * at + g], generator)[0])
                group.append(g)
                picked.append(tasks[4 * at + g])
        with torch.no_grad():
            reward = scores(examples)
        learned += reward.tolist()
        liked += [readers(words, task, one) for task, one in zip(picked, examples)]
        loss = score_loss(answer_log_chances(model, examples, words.end), reward, group)
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        if (at + 1) % 20 == 0:
            curve.append({"scorer": sum(learned) / 320, "readers": sum(liked) / 320})
            say(
                f"rounds {at - 18} to {at + 1}: the scorer gives the stories "
                f"{curve[-1]['scorer']:+.2f} on average, the readers {curve[-1]['readers']:.2f}"
            )
            learned, liked = [], []
    story = words.decode([i for i in examples[-1][0][examples[-1][1] :] if i != words.end])
    say(f"a story from the last round: {story.strip()!r}")
    save(
        "28_scorer",
        torch=torch.__version__,
        device="cpu",
        pairs={"learn": len(learn), "held": len(held)},
        ranked_right={"before": before, "after": sum(ranked), "interval": [low, high]},
        before_the_push=base,
        curve=curve,
        sample=story.strip(),
    )
