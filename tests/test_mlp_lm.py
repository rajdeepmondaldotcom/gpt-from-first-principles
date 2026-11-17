import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt.mlp_lm import MLP, batch, evaluate, sample, train
from gpt.tensors import cross_entropy, encode

TEXT = Path("data/tiny_shakespeare.txt").read_text()
VOCAB = sorted(set(TEXT))
IDS = encode(TEXT, VOCAB)


def test_matrix_multiply_by_hand():
    x = torch.tensor([2.0, 1.0])
    W = torch.tensor([[1.0, 0.0, -1.0], [0.0, 3.0, 2.0]])
    assert (x @ W).tolist() == [2.0, 3.0, 0.0]
    bent = torch.tanh(x @ W)
    assert [round(v, 3) for v in bent.tolist()] == [0.964, 0.995, 0.0]
    assert [round(v, 3) for v in (1 - bent**2).tolist()] == [0.071, 0.01, 1.0]


def test_a_lookup_is_a_one_hot_multiply():
    torch.manual_seed(0)
    model, x = MLP(65), torch.tensor([[3, 0, 64, 3, 7, 7, 1, 2]])
    one_hot = F.one_hot(x, 65).float()  # a 1 at each character's place, 0 elsewhere
    assert torch.allclose(model.embed(x), one_hot @ model.embed.weight)


def test_a_linear_layer_is_multiply_then_add():
    torch.manual_seed(0)
    layer, x = MLP(65).hidden, torch.randn(5, 8 * 24)
    assert torch.allclose(layer(x), x @ layer.weight.T + layer.bias, atol=1e-6)


def test_the_parameter_count_is_what_the_lesson_says():
    counted = sum(p.numel() for p in MLP(65).parameters())
    assert counted == 65 * 24 + (8 * 24 * 256 + 256) + (256 * 65 + 65) == 67673
    assert 65**8 > 300e12  # rows in a table that looks eight characters back


def test_a_batch_is_windows_and_what_follows_them():
    ids = torch.arange(100)
    x, y = batch(ids, 8, 32, torch.Generator().manual_seed(0))
    assert x.shape == (32, 8) and y.shape == (32,)
    assert (x[:, 1:] - x[:, :-1] == 1).all() and (y == x[:, -1] + 1).all()


def test_it_starts_with_no_opinion():
    torch.manual_seed(0)
    assert math.isclose(evaluate(MLP(65), IDS[:5000]), math.log(65), rel_tol=1e-6)


def test_the_slope_of_tanh_is_one_minus_its_square():
    for x in (0.0, 2.0, 3.0):
        nudged = (math.tanh(x + 1e-6) - math.tanh(x)) / 1e-6
        assert math.isclose(nudged, 1 - math.tanh(x) ** 2, abs_tol=1e-5)


def test_hidden_units_that_start_equal_stay_equal():
    torch.manual_seed(0)
    model = MLP(65)
    with torch.no_grad():
        model.hidden.weight[1] = model.hidden.weight[0]  # unit 1 starts as a copy of unit 0
        model.hidden.bias[1] = model.hidden.bias[0]
    train(model, IDS, steps=50, rate=0.1, generator=torch.Generator().manual_seed(0))
    assert torch.equal(model.hidden.weight[0], model.hidden.weight[1])
    assert not torch.equal(model.hidden.weight[0], model.hidden.weight[2])


def test_evaluate_is_the_loss_over_every_window():
    torch.manual_seed(0)
    model = MLP(65)
    with torch.no_grad():
        model.out.weight.normal_()  # an untrained model with opinions, so windows differ
    ids = IDS[: 4096 + 8 + 100]  # 4,196 windows: one full chunk of 4,096 and a small one
    windows = ids.unfold(0, 9, 1)
    with torch.no_grad():
        at_once = cross_entropy(model(windows[:, :-1]), windows[:, -1]).item()
    assert math.isclose(evaluate(model, ids), at_once, rel_tol=1e-5)


def test_a_bad_starting_scale_jams_the_hidden_units():
    def start(scale):
        torch.manual_seed(0)
        model = MLP(65)
        with torch.no_grad():
            model.hidden.weight *= scale
            model.out.weight.normal_(0, 0.1)  # give the loss a path back to the hidden layer
        x, y = batch(IDS, 8, 256, torch.Generator().manual_seed(0))
        h = model.hidden(model.embed(x).flatten(1)).tanh()
        cross_entropy(model(x), y).backward()
        return (h.abs() > 0.99).float().mean().item(), model.hidden.weight.grad.norm().item()

    (stuck, slopes), (stuck_big, slopes_big) = start(1), start(20)
    assert stuck < 0.01 and stuck_big > 0.5
    assert slopes > 3 * slopes_big


def test_it_beats_the_bigram():
    torch.manual_seed(0)
    n = int(len(IDS) * 0.9)
    model = MLP(65)
    train(model, IDS[:n], steps=2000, rate=0.1, generator=torch.Generator().manual_seed(0))
    assert evaluate(model, IDS[n : n + 20000]) < 2.482  # lesson 1's bigram on unseen text


def test_it_cannot_see_past_its_window():
    torch.manual_seed(0)
    model = MLP(65)
    with torch.no_grad():
        model.out.weight.normal_()
    near = encode("upon a time", VOCAB)
    a = sample(
        model, torch.cat([encode("Once ", VOCAB), near]), 5, torch.Generator().manual_seed(0)
    )
    b = sample(
        model, torch.cat([encode("Twice ", VOCAB), near]), 5, torch.Generator().manual_seed(0)
    )
    assert a[-5:] == b[-5:]  # the same last 8 characters give the same continuation
