"""Lesson 35. Speculative decoding: a small model guesses ahead and the big one checks in one pass."""

import torch

from gpt.tensors import softmax


def keep_chance(p, q):
    """The chance of keeping each token the draft might pick: 1 where the big model likes it more."""
    return (p / q).clamp(max=1)


def leftover(p, q):
    """What to draw from after a rejection: the chance the big model has and the draft lacks."""
    left = (p - q).clamp(min=0)
    return left / left.sum().clamp(min=1e-12)


def imitation_loss(draft_scores, big_scores):
    """How far the draft's chances are from the big model's: lesson 3's loss with whole rows of
    chances as targets, where it had one right token."""
    targets = softmax(big_scores)
    scores = draft_scores - draft_scores.max(dim=-1, keepdim=True).values
    log_chances = scores - scores.exp().sum(dim=-1, keepdim=True).log()
    return -(targets * log_chances).sum(dim=-1).mean()


def chances(model, ids, kept, temperature):
    """Read `ids` on top of what the cache holds. One row of chances for each token read."""
    device = model.embed.weight.device
    return softmax(model(torch.tensor([ids], device=device), kept)[0] / temperature).cpu()


def trim(kept, length):
    """Make a cache forget everything after its first `length` tokens."""
    for one in kept:
        one["k"], one["v"] = one["k"][:, :, :length], one["v"][:, :, :length]


@torch.no_grad()
def speculate(big, draft, ids, n, k=4, generator=None, temperature=1.0):
    """Write n tokens after `ids`, each drawn as the big model alone would draw it."""
    assert temperature > 0, "this draws by chance: at a temperature of 0 use lesson 22's greedy"
    ids, start = list(ids), len(ids)
    big_kept, draft_kept = [{} for _ in big.blocks], [{} for _ in draft.blocks]
    big_read = draft_read = 0  # how many tokens each cache holds
    counts = {"passes": 0, "checked": 0, "kept": 0}
    while len(ids) - start < n:
        guesses, q = [], []
        for _ in range(k):  # the draft writes k tokens, one at a time
            unread = (ids + guesses)[draft_read:]
            q.append(chances(draft, unread, draft_kept, temperature)[-1])
            draft_read += len(unread)
            guesses.append(torch.multinomial(q[-1], 1, generator=generator).item())
        # One pass of the big model: its chances after `ids` and after each guess.
        p = chances(big, (ids + guesses)[big_read:], big_kept, temperature)[-(k + 1) :]
        counts["passes"] += 1
        for j, guess in enumerate(guesses):
            counts["checked"] += 1  # a guess after a rejection is dropped unchecked
            if torch.rand((), generator=generator) < keep_chance(p[j], q[j])[guess]:
                ids.append(guess)
                counts["kept"] += 1
            else:  # the first rejection ends the round: draw this token again, drop the rest
                ids.append(torch.multinomial(leftover(p[j], q[j]), 1, generator=generator).item())
                break
        else:  # every guess kept: the big model's last row gives one more token for nothing
            ids.append(torch.multinomial(p[k], 1, generator=generator).item())
        big_read = len(ids) - 1  # the newest token is not in either cache yet
        draft_read = min(draft_read, big_read)
        trim(big_kept, big_read)
        trim(draft_kept, draft_read)
    return ids[: start + n], counts


if __name__ == "__main__":
    import time
    from pathlib import Path

    from gpt.data import read
    from gpt.generate import generate
    from gpt.gpt2 import batch, evaluate, loss_on
    from gpt.model import GPT
    from gpt.optim import AdamW
    from gpt.results import save, say
    from gpt.scale import shape
    from gpt.story import load
    from gpt.train import clip, rate_at

    cache = Path("data/cache")
    big, words = load(cache / "story")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = read(cache / "stories_train.bin").to(device)
    val_ids = read(cache / "stories_val.bin")
    steps = 1500

    def draft(imitate):
        """Train a small model on the stories, or on the big model's chances for the same text."""
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        small = GPT(**shape(96, 3)).to(device)
        optimizer = AdamW(small.parameters(), rate=0.0, decay=0.1)
        for step in range(steps):
            optimizer.rate = rate_at(step, 3e-3, steps // 10, steps)
            x, y = batch(ids, small.context, 32, generator)
            if imitate:
                with torch.no_grad():
                    target = big(x)
                loss = imitation_loss(small(x), target)
            else:
                loss = loss_on(small, x, y)
            small.zero_grad()
            loss.backward()
            clip(small.parameters(), 1.0)
            optimizer.step()
        return small.to("cpu").eval()

    big.to(device)
    drafts = {"a draft trained on the stories": draft(False)}
    drafts["a draft trained on the big model's chances"] = draft(True)
    big.to("cpu")
    count = sum(p.numel() for p in drafts["a draft trained on the stories"].parameters())
    say(f"each draft: {count:,} parameters, {steps:,} steps of 32 windows")
    for name, small in drafts.items():
        say(f"{name}: {evaluate(small, val_ids[:100_000]):.3f} nats on unseen stories")

    # The same 20 prompts, 128 tokens after each, three ways.
    prompts = [val_ids[300 * i : 300 * i + 64].tolist() for i in range(20)]
    generator = torch.Generator().manual_seed(0)
    generate(big, prompts[0], 16, generator=generator)  # a first call, to warm up
    began = time.perf_counter()
    for prompt in prompts:
        generate(big, prompt, 128, generator=generator)
    alone = 20 * 128 / (time.perf_counter() - began)
    say(f"the big model alone: {alone:,.0f} tokens a second")
    numbers = {"alone": alone}
    for name, small in drafts.items():
        total = {"passes": 0, "checked": 0, "kept": 0}
        began = time.perf_counter()
        for prompt in prompts:
            _, counts = speculate(big, small, prompt, 128, k=4, generator=generator)
            total = {key: total[key] + counts[key] for key in total}
        speed = 20 * 128 / (time.perf_counter() - began)
        share = total["kept"] / total["checked"]
        say(
            f"with {name}: {share:.0%} of the guesses it checked are kept, "
            f"{20 * 128 / total['passes']:.2f} tokens for each pass of the big model, "
            f"{speed:,.0f} tokens a second"
        )
        numbers[name] = {"kept": share, "passes": total["passes"], "tokens_a_second": speed}
    save("35_speculate", torch=torch.__version__, trained_on=device, device="cpu", numbers=numbers)
