"""Makes tests/llama_reference.*: a tiny Llama with random weights, and the scores it gives.

Run once, with Hugging Face's code as the reference:
    uv run --with transformers --with safetensors python tests/llama_reference.py
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
config = LlamaConfig(
    vocab_size=64,
    hidden_size=48,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    initializer_range=0.2,  # large enough that attention is far from even
)
model = LlamaForCausalLM(config).eval()
ids = torch.randint(64, (1, 40), generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    scores = model(ids).logits[0]
weights = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}  # tied
save_file(weights, "tests/llama_reference.safetensors")
saved = {"ids": ids[0].tolist(), "top": scores.argmax(-1).tolist(), "last": scores[-1].tolist()}
Path("tests/llama_reference.json").write_text(json.dumps(saved))
print(scores[-1].abs().max().item(), len(weights))
