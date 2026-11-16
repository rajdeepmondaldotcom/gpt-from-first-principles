"""Lessons 8 to 10. The GPT of 2019: positions, the block, and the whole model."""

import json
import math
import struct
from pathlib import Path

import torch
from torch import nn

from gpt.attention import CausalSelfAttention
from gpt.results import save, say
from gpt.tensors import cross_entropy, encode, softmax


class Embedding(nn.Module):
    """Lesson 8. A vector for what the token is, plus a vector for where it stands."""

    def __init__(self, vocab, context, width):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)  # one row per place, up to `context` places

    def forward(self, idx):
        places = torch.arange(idx.size(-1), device=idx.device)
        return self.token(idx) + self.position(places)


class LayerNorm(nn.Module):
    """Lesson 9. Give each token's vector a mean of 0 and a spread of 1, then rescale."""

    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        spread = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5).sqrt()
        return self.gain * x / spread + self.shift


def gelu(x):
    """A smooth bend: about 0 for negative numbers, about x for positive ones."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class Block(nn.Module):
    """Lesson 9. Read from the other tokens, then think, each time adding to what is there."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.norm2 = LayerNorm(width)
        self.wide = nn.Linear(width, 4 * width)
        self.narrow = nn.Linear(4 * width, width)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.narrow(gelu(self.wide(self.norm2(x))))


class GPT2(nn.Module):
    """Lesson 10. Token ids in, a row of scores for the next token at each place out."""

    def __init__(self, vocab=50257, context=1024, width=768, heads=12, layers=12):
        super().__init__()
        self.context = context  # the most tokens the model can read at once
        self.embed = Embedding(vocab, context, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = LayerNorm(width)
        self.out = nn.Linear(width, vocab, bias=False)
        self.out.weight = self.embed.token.weight  # one grid reads tokens and scores them
        for name, p in self.named_parameters():
            if p.dim() == 2:
                # Each block adds to the same vector. Smaller additions keep the sum steady.
                adds = name.endswith(("proj.weight", "narrow.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers) if adds else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


def batch(ids, context, size, generator):
    """Random windows of `context` tokens, and the same windows moved one token on."""
    at = torch.randint(len(ids) - context, (size,), generator=generator).to(ids.device)[:, None]
    steps = torch.arange(context, device=ids.device)
    return ids[at + steps], ids[at + 1 + steps]


def loss_on(model, x, y):
    """One guess per place: the scores at place i are scored against token i + 1."""
    scores = model(x)
    return cross_entropy(scores.view(-1, scores.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, ids):
    """Loss in nats over `ids`, cut into windows that do not overlap."""
    rows = (len(ids) - 1) // model.context
    assert rows > 0, "evaluate needs more than `context` tokens"
    x = ids[: rows * model.context].view(rows, model.context)
    y = ids[1 : rows * model.context + 1].view(rows, model.context)
    some = max(1, 4096 // model.context)  # rows per pass: about 4,096 tokens whatever the context
    total = sum(loss_on(model, a, b) * len(a) for a, b in zip(x.split(some), y.split(some)))
    return total.item() / rows


@torch.no_grad()
def sample(model, start, n, generator):
    """Write n more tokens. The model reads at most its last `context` tokens."""
    out, device = start.tolist(), start.device
    for _ in range(n):
        scores = model(torch.tensor([out[-model.context :]], device=device))[0, -1]
        # Draw on the CPU: the same seed then gives the same text on any machine's CPU.
        out.append(torch.multinomial(softmax(scores).cpu(), 1, generator=generator).item())
    return out


def read_safetensors(path):
    """The safetensors format: 8 bytes giving a length, that many bytes of JSON, then raw numbers."""
    data = Path(path).read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    tensors = {}
    for name, entry in header.items():
        if name != "__metadata__":
            assert entry["dtype"] == "F32", f"{name} is {entry['dtype']}, not 32-bit floats"
            start, end = entry["data_offsets"]
            raw = bytearray(data[8 + length + start : 8 + length + end])
            tensors[name] = torch.frombuffer(raw, dtype=torch.float32).view(entry["shape"])
    return tensors


def load_gpt2(path):
    """Build our model and pour OpenAI's released GPT-2 weights into it."""
    theirs = read_safetensors(path)
    model = GPT2()
    ours = {"embed.token.weight": "wte.weight", "embed.position.weight": "wpe.weight"}
    ours |= {"norm.gain": "ln_f.weight", "norm.shift": "ln_f.bias"}
    for i in range(len(model.blocks)):
        for mine, name in [
            ("norm1.gain", "ln_1.weight"),
            ("norm1.shift", "ln_1.bias"),
            ("attn.qkv.weight", "attn.c_attn.weight"),
            ("attn.qkv.bias", "attn.c_attn.bias"),
            ("attn.proj.weight", "attn.c_proj.weight"),
            ("attn.proj.bias", "attn.c_proj.bias"),
            ("norm2.gain", "ln_2.weight"),
            ("norm2.shift", "ln_2.bias"),
            ("wide.weight", "mlp.c_fc.weight"),
            ("wide.bias", "mlp.c_fc.bias"),
            ("narrow.weight", "mlp.c_proj.weight"),
            ("narrow.bias", "mlp.c_proj.bias"),
        ]:
            ours[f"blocks.{i}.{mine}"] = f"h.{i}.{name}"
    state = {}
    for mine, name in ours.items():
        tensor = theirs[name]
        # OpenAI stored each linear layer as (inputs, outputs). PyTorch wants (outputs, inputs).
        flip = mine.endswith(("qkv.weight", "proj.weight", "wide.weight", "narrow.weight"))
        state[mine] = tensor.T if flip else tensor
    state["out.weight"] = state["embed.token.weight"]  # the tied grid, under its second name
    model.load_state_dict(state)  # raises if any weight is missing or unexpected
    return model.eval()


if __name__ == "__main__":
    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    n = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:n], ids[n:]

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = GPT2(vocab=len(vocab), context=64, width=128, heads=4, layers=4)
    parameters = sum(p.numel() for p in model.parameters())
    first = evaluate(model, val_ids)
    say(f"parameters: {parameters:,}")
    say(f"loss before training: {first:.3f} nats (ln 65 = {math.log(65):.3f})")
    for step in range(1, 3001):
        x, y = batch(train_ids, model.context, 32, generator)
        loss = loss_on(model, x, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= 0.1 * p.grad
        if step % 500 == 0:
            say(f"step {step}: {loss.item():.3f} nats")
    val_nats = evaluate(model, val_ids)
    story = sample(model, encode("Once upon a time", vocab), 80, generator)
    story = "".join(vocab[i] for i in story)
    say(f"unseen text: {val_nats:.3f} nats, {val_nats / math.log(2):.3f} bits per byte")
    say(f"sample: {story!r}")
    save(
        "10_gpt2",
        torch=torch.__version__,
        device="cpu",
        seed=0,
        parameters=parameters,
        first_nats=first,
        val_nats=val_nats,
        val_bits_per_byte=val_nats / math.log(2),
        sample=story,
    )
