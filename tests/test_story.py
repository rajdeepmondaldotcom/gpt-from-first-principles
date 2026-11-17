import torch

from gpt.gpt2 import read_safetensors
from gpt.model import GPT, load_llama
from gpt.story import save_llama


def model(seed):
    torch.manual_seed(seed)
    return GPT(vocab=64, context=16, width=48, heads=4, layers=2, kv_heads=2)


def test_saved_weights_load_back_the_same(tmp_path):
    first, second = model(0), model(1)
    save_llama(first, tmp_path / "model.safetensors")
    load_llama(second, tmp_path / "model.safetensors")
    ids = torch.randint(64, (1, 16))
    assert torch.equal(first(ids), second(ids))


def test_the_file_has_the_names_and_shapes_of_the_reference_llama(tmp_path):
    save_llama(model(0), tmp_path / "model.safetensors")
    ours = read_safetensors(tmp_path / "model.safetensors")
    theirs = read_safetensors("tests/llama_reference.safetensors")
    assert {name: t.shape for name, t in ours.items()} == {
        name: t.shape for name, t in theirs.items()
    }
