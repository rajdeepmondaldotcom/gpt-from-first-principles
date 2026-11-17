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
    before[0] = -1  # the likeliest token always stays
    chances[order[before >= top_p]] = 0  # the likelier ones already reach top_p: drop this one
    return torch.multinomial(chances.cpu(), 1, generator=generator).item()


@torch.no_grad()
def generate(model, ids, n, stop=None, cache=True, **how):
    """Write up to n tokens after `ids`, stopping at the context the model trained on.

    With a cache, the model reads each token once.
    """
    ids, device = list(ids), model.embed.weight.device
    kept = [{} for _ in model.blocks] if cache else None
    unread = ids
    for _ in range(min(n, model.context - len(ids))):
        scores = model(torch.tensor([unread if cache else ids], device=device), kept)[0, -1]
        ids.append(choose(scores, **how))
        if ids[-1] == stop:
            break
        unread = ids[-1:]
    return ids


if __name__ == "__main__":
    from gpt.cost import cache_bytes
    from gpt.data import documents, read
    from gpt.story import load

    # On the processor: one token at a time is faster there than on the graphics chip, and it repeats.
    model, words = load(Path("data/cache/story"))
    prompt = words.encode("Once upon a time")
    known = set()  # each word of the training text, in lowercase
    for story in documents(read(Path("data/cache/stories_train.bin")), words.end):
        known.update(re.findall(r"[a-z]+", words.decode(story.tolist()).lower()))

    def made_up(texts):
        """The share of words in these texts that the training text never uses."""
        used = [word for text in texts for word in re.findall(r"[a-z]+", text.lower())]
        return sum(word not in known for word in used) / len(used)

    def write(**how):
        """One story as text, and whether it reached its end."""
        ids = generate(model, prompt, 250, stop=words.end, **how)
        return words.decode([i for i in ids if i != words.end]), ids[-1] == words.end

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
        finished = sum(ended for _, ended in stories)
        invented = made_up([text for text, _ in stories])
        settings[name] = {"different": different, "finished": finished, "made_up": invented}
        samples[name] = stories[0][0]
        say(
            f"{name}: {different} different stories of 20, {finished} reach their end, "
            f"{invented:.1%} of their words made up"
        )
        say(f"  {stories[0][0][:150]!r}")
    save("22_sampling", torch=torch.__version__, device="cpu", settings=settings, samples=samples)

    # Lesson 23: the same story with and without the cache.
    counted = []
    hook = model.embed.register_forward_hook(
        lambda part, inputs, out: counted.append(inputs[0].numel())
    )
    timings = {}
    for cache in (False, True):
        counted.clear()
        began = time.perf_counter()
        story = generate(model, prompt, 200, temperature=0, cache=cache)
        took = time.perf_counter() - began
        timings[cache] = {"tokens_read": sum(counted), "seconds": took, "story": story}
        name = "with the cache" if cache else "without a cache"
        say(f"{name}: {sum(counted):,} tokens read, {200 / took:.0f} tokens a second")
    hook.remove()
    same = timings[True]["story"] == timings[False]["story"]
    say(f"the two stories are the same: {same}")
    say(f"the cache holds {cache_bytes(model, 1):,} bytes for each token")
    save(
        "23_cache",
        torch=torch.__version__,
        device="cpu",
        read={str(k): v["tokens_read"] for k, v in timings.items()},
        seconds={str(k): v["seconds"] for k, v in timings.items()},
        same=same,
    )
