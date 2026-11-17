"""Lessons 22 and 23. Writing: choose each next token, and never read a token twice."""

import re
import time
from pathlib import Path

import torch

from gpt.results import save, say
from gpt.tensors import softmax


def choose(scores, temperature=1.0, top_p=1.0, generator=None):
    """Pick the next token from one row of scores."""
    if temperature == 0:
        return scores.argmax().item()  # no chance involved: the top score wins
    chances = softmax(scores / temperature)  # under 1 sharpens the row, over 1 flattens it
    order = chances.argsort(descending=True)
    before = chances[order].cumsum(0) - chances[order]  # the total chance of the likelier tokens
    chances[order[before >= top_p]] = 0  # the likelier ones already reach top_p: drop this one
    return torch.multinomial(chances.cpu(), 1, generator=generator).item()


@torch.no_grad()
def generate(model, ids, n, stop=None, cache=True, **how):
    """Write up to n tokens after `ids`. With a cache, the model reads each token once."""
    ids, device = list(ids), model.embed.weight.device
    kept = [{} for _ in model.blocks] if cache else None
    unread = ids
    for _ in range(n):
        scores = model(torch.tensor([unread if cache else ids], device=device), kept)[0, -1]
        ids.append(choose(scores, **how))
        if ids[-1] == stop:
            break
        unread = ids[-1:]
    return ids


if __name__ == "__main__":
    from gpt.cost import cache_bytes
    from gpt.data import read
    from gpt.story import load

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, words = load(Path("data/cache/story"), device)
    prompt = words.encode("Once upon a time")
    # Each lowercase word in 5 million tokens of the training text: the words the stories use.
    trained = words.decode(read(Path("data/cache/stories_train.bin"))[:5_000_000].tolist())
    known = set(re.findall(r"[a-z]+", trained.lower()))

    def made_up(texts):
        """The share of words in these texts that the training text never uses."""
        used = [word for text in texts for word in re.findall(r"[a-z]+", text.lower())]
        return sum(word not in known for word in used) / len(used)

    def write(n=250, **how):
        """One story as text, and how many tokens it took."""
        ids = generate(model, prompt, n, stop=words.end, **how)
        return words.decode([i for i in ids if i != words.end]), len(ids) - len(prompt)

    # Lesson 22: the same prompt, twenty times, at each setting.
    generator, settings, samples = torch.Generator().manual_seed(0), {}, {}
    for name, how in [
        ("temperature 0", {"temperature": 0}),
        ("temperature 0.5", {"temperature": 0.5}),
        ("temperature 1", {"temperature": 1.0}),
        ("temperature 1.5", {"temperature": 1.5}),
        ("temperature 1, top-p 0.9", {"temperature": 1.0, "top_p": 0.9}),
    ]:
        stories = [write(generator=generator, **how) for _ in range(20)]
        different = len({text for text, _ in stories})
        finished = sum(count < 250 for _, count in stories)
        invented = made_up([text for text, _ in stories])
        settings[name] = {"different": different, "finished": finished, "made_up": invented}
        samples[name] = stories[0][0]
        say(
            f"{name}: {different} different stories of 20, {finished} reach their end, "
            f"{invented:.1%} of their words made up"
        )
        say(f"  {stories[0][0][:150]!r}")
    save("22_sampling", torch=torch.__version__, device=device, settings=settings, samples=samples)

    # Lesson 23: the same story with and without the cache.
    read = []
    hook = model.embed.register_forward_hook(
        lambda part, inputs, out: read.append(inputs[0].numel())
    )
    timings = {}
    for cache in (False, True):
        read.clear()
        began = time.perf_counter()
        story = generate(model, prompt, 200, temperature=0, cache=cache)
        took = time.perf_counter() - began
        timings[cache] = {"tokens_read": sum(read), "seconds": took, "story": story}
        name = "with the cache" if cache else "without a cache"
        say(f"{name}: {sum(read):,} tokens read, {200 / took:.0f} tokens a second")
    hook.remove()
    same = timings[True]["story"] == timings[False]["story"]
    say(f"the two stories are the same: {same}")
    say(f"the cache holds {cache_bytes(model, 1):,} bytes for each token")
    save(
        "23_cache",
        torch=torch.__version__,
        device=device,
        read={str(k): v["tokens_read"] for k, v in timings.items()},
        seconds={str(k): v["seconds"] for k, v in timings.items()},
        same=same,
    )
