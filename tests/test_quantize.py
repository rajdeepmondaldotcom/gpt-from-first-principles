import pytest
import torch

from gpt.model import GPT
from gpt.quantize import Quantized, pack, quantize, shrink, unpack


def test_one_row_by_hand():
    numbers, scale = quantize(torch.tensor([[0.6, -1.0, 0.26, 0.0]]), bits=4)
    assert scale.item() == pytest.approx(1 / 7)  # the largest size, 1.0, maps to 7
    assert numbers.tolist() == [[4, -7, 2, 0]]  # 4.2 rounds to 4, 1.82 rounds to 2
    back = numbers * scale
    assert back[0].tolist() == pytest.approx([4 / 7, -1.0, 2 / 7, 0.0])


@pytest.mark.parametrize("bits", [8, 4])
def test_no_weight_moves_by_more_than_half_a_step(bits):
    grid = torch.randn(64, 48, generator=torch.Generator().manual_seed(0)) * 0.02
    numbers, scale = quantize(grid, bits)
    assert ((numbers * scale - grid).abs() <= scale / 2 + 1e-9).all()
    assert numbers.abs().max() <= 2 ** (bits - 1) - 1


def test_each_row_has_a_scale_of_its_own():
    numbers, scale = quantize(torch.tensor([[1.0, -0.25], [0.001, 0.00025]]), bits=8)
    assert scale.shape == (2, 1) and numbers.tolist() == [[127, -32], [127, 32]]
    # One scale for the whole grid would turn the small row into zeros.


def test_a_row_of_zeros_stays_zero():
    numbers, scale = quantize(torch.zeros(2, 4), bits=8)
    assert (numbers == 0).all() and (scale > 0).all()  # a scale of 0 would divide zero by zero
    assert (numbers * scale == 0).all()


def test_packing_two_to_a_byte_comes_back_the_same():
    numbers = torch.randint(-7, 8, (6, 10), generator=torch.Generator().manual_seed(0)).to(
        torch.int8
    )
    packed = pack(numbers)
    assert packed.numel() == 30 and packed.dtype == torch.uint8
    assert torch.equal(unpack(packed, numbers.shape), numbers)


def test_a_layers_output_moves_by_no_more_than_the_bound():
    torch.manual_seed(0)
    layer, x = torch.nn.Linear(48, 64, bias=False), torch.randn(5, 48)
    for bits in (8, 4):
        small = Quantized(layer, bits)
        # Each weight is off by at most half a step, so a row's output is off by at most
        # half a step times the sum of the sizes of the inputs.
        bound = small.scale.T / 2 * x.abs().sum(dim=1, keepdim=True)
        assert ((small(x) - layer(x)).abs() <= bound + 1e-6).all()


def test_the_memory_falls_four_times_and_eight_times():
    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 256, bias=False)
    full = layer.weight.nbytes
    assert Quantized(layer, 8).numbers.nbytes == full / 4
    assert Quantized(layer, 4).numbers.nbytes == full / 8
    assert Quantized(layer, 8).scale.nbytes == 256 * 4  # one scale a row: 1.6% on top at 8 bits


def test_a_shrunk_model_gives_nearly_the_same_scores_and_closer_at_8_bits():
    torch.manual_seed(0)
    ids = torch.randint(65, (2, 16))
    gaps = {}
    for bits in (8, 4):
        torch.manual_seed(1)
        model = GPT(vocab=65, context=16, width=64, heads=4, layers=2, kv_heads=2)
        with torch.no_grad():
            before = model(ids)
            gaps[bits] = (shrink(model, bits)(ids) - before).abs().max().item()
        assert not any(isinstance(m, torch.nn.Linear) for b in model.blocks for m in b.modules())
    assert gaps[8] < gaps[4] < 0.1 and gaps[8] < 0.005
