"""Make tests/gpt2_reference.json from Hugging Face's own GPT-2 code. Run once:

PYTHONPATH=. uv run --with transformers --extra reference python tests/gpt2_reference.py
"""

import json
from pathlib import Path

import tiktoken
import torch
from safetensors.torch import load_file  # their reader, not ours: the two sides stay independent
from transformers import GPT2Config, GPT2LMHeadModel

weights = load_file("data/cache/gpt2.safetensors")
model = GPT2LMHeadModel(GPT2Config()).eval()
state = {f"transformer.{name}": tensor for name, tensor in weights.items()}
state["lm_head.weight"] = weights["wte.weight"]
missing = model.load_state_dict(state, strict=False).missing_keys
assert not missing, missing

ids = tiktoken.get_encoding("gpt2").encode("Once upon a time, there was a")
with torch.no_grad():
    scores = model(torch.tensor([ids])).logits[0]
check = sorted({*scores[-1].topk(20).indices.tolist(), *range(0, 50257, 5000)})
reference = {
    "prompt": "Once upon a time, there was a",
    "ids": ids,
    "argmax": scores.argmax(dim=-1).tolist(),
    "check": check,
    "last": [round(scores[-1, i].item(), 5) for i in check],
}
Path("tests/gpt2_reference.json").write_text(json.dumps(reference) + "\n")
print(reference["argmax"], tiktoken.get_encoding("gpt2").decode(reference["argmax"]))
