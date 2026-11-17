import json

import pytest
import torch

from gpt.export import export, mlx_scores
from gpt.model import GPT
from gpt.story import save_llama

SHAPE = {"vocab": 64, "context": 32, "width": 48, "heads": 4, "kv_heads": 2, "layers": 2}


def saved(tmp_path):
    torch.manual_seed(0)
    model = GPT(**SHAPE)
    save_llama(model, tmp_path / "model.safetensors")
    (tmp_path / "shape.json").write_text(json.dumps(SHAPE))
    export(tmp_path, tmp_path / "out")
    return model


def test_the_config_names_each_size(tmp_path):
    saved(tmp_path)
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["hidden_size"] == 48 and config["num_hidden_layers"] == 2
    assert config["num_attention_heads"] == 4 and config["num_key_value_heads"] == 2
    assert config["intermediate_size"] == 128 and config["vocab_size"] == 64  # 8/3 of 48 is 128
    assert (tmp_path / "out" / "model.safetensors").read_bytes() == (
        tmp_path / "model.safetensors"
    ).read_bytes()


@pytest.mark.reference
def test_apples_engine_gives_our_scores(tmp_path):
    pytest.importorskip("mlx_lm")  # Apple's engine: Macs only
    model = saved(tmp_path)
    ids = torch.randint(64, (2, 32), generator=torch.Generator().manual_seed(0))
    ours, theirs = model(ids), mlx_scores(tmp_path / "out", ids.tolist())
    assert torch.allclose(ours, theirs, atol=1e-4)
    assert (ours.argmax(-1) == theirs.argmax(-1)).all()
