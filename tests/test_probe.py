import pytest
import torch

chess = pytest.importorskip("chess")  # these tests need the chess extra

from gpt.bpe import Tokenizer, train
from gpt.model import GPT
from gpt.probe import ends, erase, fit, lens, read, squares, streams

WORDS = Tokenizer(train(" e4 e5 Nf3 Nc6 Bb5 a6 O-O Nxe4" * 40, 300))


def model():
    torch.manual_seed(0)
    return GPT(vocab=WORDS.end + 1, context=64, width=32, heads=4, layers=3)


def test_squares_by_hand():
    start = squares([])
    assert start[:8] == [4, 2, 3, 5, 6, 3, 2, 4]  # white's back row, a1 to h1: rook, knight, ...
    assert start[8:16] == [1] * 8 and start[16:48] == [0] * 32 and start[48:56] == [7] * 8
    after = squares(["e4", "e5", "Nf3"])
    assert after[12] == 0 and after[28] == 1  # the pawn left e2 for e4
    assert after[36] == 7 and after[21] == 2 and after[6] == 0  # e5, and the knight on f3


def test_the_last_stream_through_the_lens_is_the_models_scores():
    net, ids = model(), [WORDS.end, *WORDS.encode(" e4 e5 Nf3")]
    seen = streams(net, ids)
    assert len(seen) == 3 and seen[0].shape == (len(ids), 32)
    assert torch.allclose(lens(net, seen[-1]), net(torch.tensor([ids]))[0], atol=1e-6)


def test_an_edit_changes_the_last_token_from_that_block_on_and_nothing_else():
    net, ids = model(), [WORDS.end, *WORDS.encode(" e4 e5 Nf3")]
    plain, change = streams(net, ids), torch.ones(32)
    edited = streams(net, ids, edit=(1, change))
    assert torch.equal(edited[0], plain[0])  # the block before the edit
    assert torch.allclose(edited[1][-1], plain[1][-1] + change)
    assert torch.equal(edited[2][:-1], plain[2][:-1])  # earlier tokens never read a later one
    assert not torch.allclose(edited[2][-1], plain[2][-1])


def test_each_moves_last_token_is_found():
    played = ["e4", "e5", "O-O"]
    places = ends(WORDS, played)
    ids = [WORDS.end, *WORDS.encode(" e4 e5 O-O")]
    assert places[-1] == len(ids) - 1 and len(places) == 3
    for move, place in zip(played, places):
        assert WORDS.decode(ids[1 : place + 1]).endswith(move)


def test_a_probe_reads_what_is_written_in_a_straight_line_and_not_shuffled_labels():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(3000, 16, generator=generator)
    # A made-up rule: the piece on square s is set by the sign of two numbers of the stream.
    y = torch.stack(
        [(x[:, s % 16] > 0).long() + 2 * (x[:, (s + 5) % 16] > 0) for s in range(64)], 1
    )
    probe = fit(x[:2500], y[:2500], generator, steps=600)
    right = (read(probe, x[2500:]) == y[2500:]).float().mean().item()
    mixed = y[torch.randperm(2500, generator=generator)]  # the same boards, on the wrong streams
    control = fit(x[:2500], mixed, generator, steps=600)
    wrong = (read(control, x[2500:]) == y[2500:]).float().mean().item()
    assert right > 0.9 and wrong < 0.4  # four pieces, even odds: a guess is right 1 time in 4


def test_erasing_a_piece_makes_the_probe_read_the_square_as_empty():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 16, generator=generator)
    y = torch.stack([(x[:, s % 16] > 0).long() for s in range(64)], 1)  # piece 1 or empty
    probe = fit(x, y, generator, steps=400)
    stream = x[0].clone()
    square = int((read(probe, stream[None])[0] == 1).nonzero()[0])  # a square it reads as piece 1
    change = erase(probe, stream, square, piece=1)
    after = read(probe, (stream + change)[None])[0]
    assert after[square] == 0  # now read as empty
    lead = probe(stream + change).view(64, 13)[square]
    assert (lead[0] - lead[1]).item() == pytest.approx(3.0, abs=1e-3)  # by the margin asked for
    assert change.norm() < (stream * 2).norm()  # a nudge, not a new stream
    # Where "empty" already leads by more than the margin, there is nothing to rub out.
    assert erase(probe, stream + change, square, piece=1, margin=1.0).abs().max() == 0
    # A nudge of a set length runs along the same line, whatever the probe reads.
    set_length = erase(probe, stream + change, square, piece=1, length=2.0)
    assert set_length.norm().item() == pytest.approx(2.0)
    assert torch.allclose(set_length / 2.0, change / change.norm(), atol=1e-5)
