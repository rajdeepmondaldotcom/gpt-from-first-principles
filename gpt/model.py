"""Lessons 15 and 16. Today's model: rotary positions, RMSNorm, SwiGLU, grouped heads."""

import math
from pathlib import Path

import torch
from torch import nn

from gpt.attention import attend
from gpt.gpt2 import gelu, read_safetensors
from gpt.results import save, say
from gpt.rope import rope


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads=None):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads or heads
        size = width // heads  # the width of one head
        self.q = nn.Linear(width, heads * size, bias=False)
        self.k = nn.Linear(width, self.kv_heads * size, bias=False)  # fewer key and value heads
        self.v = nn.Linear(width, self.kv_heads * size, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x, cache=None):
        """x has shape (batch, tokens, width). `cache` holds the keys and values of earlier tokens."""
        B, T, _ = x.shape
        seen = cache["k"].size(2) if cache else 0  # tokens read before this call
        pos = torch.arange(seen, seen + T, device=x.device)
        q = rope(self.q(x).view(B, T, self.heads, -1).transpose(1, 2), pos)
        k = rope(self.k(x).view(B, T, self.kv_heads, -1).transpose(1, 2), pos)
        v = self.v(x).view(B, T, self.kv_heads, -1).transpose(1, 2)
        if cache is not None:
            if cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        share = self.heads // self.kv_heads  # query heads that read each key head
        k, v = k.repeat_interleave(share, dim=1), v.repeat_interleave(share, dim=1)
        # Row i is new token i. It may read every earlier token and the new ones up to itself.
        past = torch.ones(T, seen + T, dtype=torch.bool, device=x.device).tril(seen)
        out, _ = attend(q, k, v, past)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, x):
        """Divide each vector by the root of the mean of its squares. No mean is subtracted."""
        return self.gain * x / (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = 8 * width // 3  # three grids this wide cost what lesson 9's two cost at 4 x width
        self.gate = nn.Linear(width, hidden, bias=False)
        self.wide = nn.Linear(width, hidden, bias=False)
        self.narrow = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate = self.gate(x)
        return self.narrow(gate * torch.sigmoid(gate) * self.wide(x))


class MLP(nn.Module):
    """Lesson 9's MLP as a part of its own, so an experiment can put it back."""

    def __init__(self, width):
        super().__init__()
        self.wide, self.narrow = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, x):
        return self.narrow(gelu(self.wide(x)))


class Block(nn.Module):
    def __init__(self, width, heads, kv_heads, norm, mlp):
        super().__init__()
        self.norm1, self.attn = norm(width), Attention(width, heads, kv_heads)
        self.norm2, self.mlp = norm(width), mlp(width)

    def forward(self, x, cache=None):
        x = x + self.attn(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(
        self, vocab, context, width, heads, layers, kv_heads=None, norm=RMSNorm, mlp=SwiGLU
    ):
        super().__init__()
        self.context = context  # the window it trains on; nothing in the model has this size
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, heads, kv_heads, norm, mlp) for _ in range(layers))
        self.norm = norm(width)
        self.out = nn.Linear(width, vocab, bias=False)
        self.out.weight = self.embed.weight
        for name, p in self.named_parameters():  # lesson 10's starting weights
            if p.dim() > 1:
                last = name.endswith(("proj.weight", "narrow.weight"))  # adds into the stream
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers) if last else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, cache=None):
        """Token ids in, scores out. `cache` is one dictionary per block, or None."""
        x = self.embed(idx)
        for block, kept in zip(self.blocks, cache or [None] * len(self.blocks)):
            x = block(x, kept)
        return self.out(self.norm(x))


# Our names for the parts, beside the names a Llama-format file gives them.
LLAMA = {
    "model.embed_tokens": "embed",
    "model.norm.weight": "norm.gain",
    "model.layers": "blocks",
    "self_attn.q_proj": "attn.q",
    "self_attn.k_proj": "attn.k",
    "self_attn.v_proj": "attn.v",
    "self_attn.o_proj": "attn.proj",
    "mlp.gate_proj": "mlp.gate",
    "mlp.up_proj": "mlp.wide",
    "mlp.down_proj": "mlp.narrow",
    "input_layernorm.weight": "norm1.gain",
    "post_attention_layernorm.weight": "norm2.gain",
}


def load_llama(model, path):
    """Copy the weights of a Llama-format file into `model`, name by name."""
    # shortcut: the file must fit this model: SwiGLU 8/3 of the width, one tied grid for tokens
    # and scores, 1e-6 in RMSNorm, a rotary base of 10000. Meta's released Llamas differ in each.
    state = {}
    for name, tensor in read_safetensors(path).items():
        for theirs, ours in LLAMA.items():
            name = name.replace(theirs, ours)
        state[name] = tensor
    state["out.weight"] = state["embed.weight"]  # one grid reads tokens and scores them
    model.load_state_dict(state)


if __name__ == "__main__":
    from gpt.experiment import compare, trial
    from gpt.gpt2 import LayerNorm, evaluate, sample
    from gpt.optim import AdamW
    from gpt.tensors import encode
    from gpt.train import train

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    n = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:n], ids[n:]
    shape = {"vocab": len(vocab), "context": 64, "width": 128, "heads": 4, "layers": 4}
    old = {"norm": LayerNorm, "mlp": MLP}  # lesson 15's model: rotary attention, lesson 9's parts

    def losses(**parts):
        return [
            trial(lambda: GPT(**shape, **parts), train_ids, val_ids, seed, steps=600)
            for seed in range(5)
        ]

    def count(**parts):
        return sum(p.numel() for p in GPT(**shape, **parts).parameters())

    base, runs = losses(**old), {}
    say("lesson 15's model, five seeds: " + " ".join(f"{x:.4f}" for x in base))
    for name, parts in [
        ("RMSNorm in place of layer normalization", old | {"norm": RMSNorm}),
        ("SwiGLU in place of the MLP", old | {"mlp": SwiGLU}),
        ("2 key heads in place of 4", old | {"kv_heads": 2}),
        ("all three", {"kv_heads": 2}),
    ]:
        mean, error = compare(base, losses(**parts))
        say(
            f"{name}: {mean:+.4f} nats, error {error:.4f}, ratio {abs(mean) / error:.1f}, "
            f"{count(**parts):,} parameters"
        )
        runs[name] = {"difference": mean, "error": error, "parameters": count(**parts)}

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = GPT(**shape, kv_heads=2)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    for _ in train(model, optimizer, train_ids, 5000, peak=3e-3, generator=generator):
        pass
    val = evaluate(model, val_ids)
    story = sample(model, encode("Once upon a time", vocab), 120, generator)
    story = "".join(vocab[i] for i in story)
    say(f"lesson 15's model: {count(**old):,} parameters")
    say(f"unseen text: {val:.3f} nats, {val / math.log(2):.3f} bits per byte")
    say(f"sample: {story!r}")
    save(
        "16_model",
        torch=torch.__version__,
        device="cpu",
        base=base,
        runs=runs,
        val_nats=val,
        val_bits_per_byte=val / math.log(2),
        sample=story,
    )
