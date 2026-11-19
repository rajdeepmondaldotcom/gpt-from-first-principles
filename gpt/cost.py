"""Lesson 17. What it costs: count the memory and the arithmetic before spending them."""

import time

import torch

from gpt.attention import attend
from gpt.gpt2 import loss_on
from gpt.results import save, say


def forward_flops(model, tokens, context):
    """Multiplies and adds in a forward pass over `tokens` tokens, read in windows of `context`."""
    grids = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    width, layers = model.embed.weight.size(1), len(model.blocks)
    weights = 2 * grids  # each token meets each weight once: one multiply and one add
    reading = 4 * context * width * layers  # each token scores every key, then mixes every value
    return (weights + reading) * tokens


def cache_bytes(model, tokens):
    """What the model's cache holds for `tokens` tokens: a key and a value in each block."""
    attn = model.blocks[0].attn
    return 2 * len(model.blocks) * attn.k.out_features * 4 * tokens  # 4 bytes a number


def kept_bytes(model, x, y):
    """The bytes one training step keeps from its forward pass for its backward pass."""
    own = {p.untyped_storage().data_ptr() for p in model.parameters()}
    kept = {}

    def note(tensor):
        block = tensor.untyped_storage()  # the memory a tensor's numbers sit in
        if block.data_ptr() not in own:  # the weights are counted with the training state
            kept[block.data_ptr()] = block.nbytes()  # once, however many tensors share it
        return tensor

    # PyTorch calls `note` with each tensor it saves for the backward pass.
    with torch.autograd.graph.saved_tensors_hooks(note, lambda tensor: tensor):
        loss_on(model, x, y)
    return sum(kept.values())


def attend_in_tiles(q, k, v, mask, tile):
    """Lesson 6's `attend`, `tile` queries at a time.

    With no backward pass to come, the whole grid of scores never exists at once. Under autograd
    each tile's weights are kept for the backward pass, so training holds the whole grid after all.
    """
    rows = range(0, q.size(-2), tile)
    return torch.cat(
        [attend(q[..., i : i + tile, :], k, v, mask[i : i + tile])[0] for i in rows], -2
    )


if __name__ == "__main__":
    from gpt.model import GPT
    from gpt.optim import AdamW

    def measure(device, **shape):
        """Time a forward pass and a training step on a batch of 32 windows of random tokens."""
        torch.manual_seed(0)
        model = GPT(**shape).to(device)
        optimizer = AdamW(model.parameters(), rate=1e-3)
        ids = torch.randint(shape["vocab"], (32, shape["context"] + 1), device=device)
        x, y = ids[:, :-1], ids[:, 1:]

        def forward():
            with torch.no_grad():
                model(x).sum().item()  # asking for a number waits for the graphics chip to finish

        def step():
            model.zero_grad()
            loss = loss_on(model, x, y)
            loss.backward()
            optimizer.step()
            loss.item()

        def seconds(work, repeats=10):
            """The fastest of `repeats` timings: the one the rest of the machine disturbed least."""
            best = float("inf")
            for _ in range(repeats):
                began = time.perf_counter()
                work()
                best = min(best, time.perf_counter() - began)
            return best

        return model, x, y, seconds(forward), seconds(step)

    model, x, y, forward, whole = measure("cpu", vocab=65, context=64, width=128, heads=4, layers=4)
    count = sum(p.numel() for p in model.parameters())
    flops = forward_flops(model, x.numel(), model.context)
    kept = kept_bytes(model, x, y)
    longer = torch.randint(65, (32, 129))  # the same batch with twice the context
    grown = kept_bytes(model, longer[:, :-1], longer[:, 1:]) / kept
    say(f"parameters: {count:,}, {4 * count / 1e6:.1f} MB of weights")
    say(f"training state: weights, slopes and two averages, {16 * count / 1e6:.1f} MB")
    say(f"kept for the backward pass, one batch of 32 x 64 tokens: {kept / 1e6:.1f} MB")
    say(f"the same batch with 128 tokens in each window: {grown:.2f} times as much kept")
    say(f"forward pass over 2,048 tokens: {flops / 1e9:.2f} billion multiplies and adds")
    say(f"forward pass: {forward * 1e3:.0f} ms, {flops / forward / 1e9:.0f} billion a second")
    say(f"training step: {whole * 1e3:.0f} ms, {whole / forward:.1f} times the forward pass")
    read = 4 * count + cache_bytes(model, 64)
    say(f"cache: {cache_bytes(model, 1):,} bytes a token")
    say(f"one new token after 64: {forward_flops(model, 1, 64) / read:.2f} for each byte read")
    say(f"one training step: {3 * flops / (16 * count + kept):,.0f} for each byte it holds")

    timings = {"cpu": {"forward": forward, "step": whole}}
    if torch.backends.mps.is_available():
        big, x, y, forward, whole = measure(
            "mps", vocab=4097, context=256, width=256, heads=8, kv_heads=4, layers=6
        )
        speed = 3 * forward_flops(big, x.numel(), big.context) / whole
        count = sum(p.numel() for p in big.parameters())
        say(f"on the graphics chip, {count:,} parameters: {whole * 1e3:.0f} ms a training step")
        say(f"that step takes {whole / forward:.1f} times its forward pass")
        say(f"that is {speed / 1e9:.0f} billion multiplies and adds a second")
        say(f"three hours at that speed: {3 * 3600 * speed:.1e} multiplies and adds")
        timings["mps"] = {"forward": forward, "step": whole, "flops_per_second": speed}
    save("17_cost", torch=torch.__version__, timings=timings)
