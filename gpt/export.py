"""Lesson 37. A real engine: hand our model to Apple's MLX and check that it computes the same."""

import json
import shutil

from gpt.gpt2 import read_safetensors


def export(folder, out):
    """Make a folder that engines for Llama-format models can load: our weights file as it is,
    and a config file that names each size the way those engines expect."""
    out.mkdir(parents=True, exist_ok=True)
    shape = json.loads((folder / "shape.json").read_text())
    shutil.copy(folder / "model.safetensors", out / "model.safetensors")
    wide = read_safetensors(folder / "model.safetensors")["model.layers.0.mlp.gate_proj.weight"]
    config = {
        "model_type": "llama",
        "vocab_size": shape["vocab"],
        "hidden_size": shape["width"],
        "num_hidden_layers": shape["layers"],
        "num_attention_heads": shape["heads"],
        "num_key_value_heads": shape["kv_heads"] or shape["heads"],
        "intermediate_size": wide.size(0),  # the width inside SwiGLU
        "max_position_embeddings": shape["context"],
        "rms_norm_eps": 1e-6,  # the three numbers lesson 16 fixed
        "rope_theta": 10000,
        "tie_word_embeddings": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def mlx_scores(folder, ids):
    """The scores Apple's engine gives a batch of token ids, as a PyTorch tensor."""
    import mlx.core as mx
    import torch
    from mlx_lm.utils import load_model

    return torch.tensor(load_model(folder)[0](mx.array(ids)).tolist())
