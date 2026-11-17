import pytest
import torch

chess = pytest.importorskip("chess")  # these tests need the chess extra

from gpt.bpe import Tokenizer, train
from gpt.model import GPT
from gpt.play import (
    expected,
    grabber,
    lead,
    match,
    model_mover,
    play,
    random_mover,
    rate,
)


def board_after(*moves):
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    return board


def scripted(*moves):
    """A player that plays these moves in order."""
    line = iter(moves)
    return lambda board, played: next(line)


def test_a_mate_is_a_win_for_the_side_that_gave_it():
    fool = play(scripted("f3", "g4"), scripted("e5", "Qh4#"))  # the shortest game there is
    assert fool == 0.0  # white is mated
    scholar = play(scripted("e4", "Bc4", "Qh5", "Qxf7#"), scripted("e5", "Nc6", "Nf6"))
    assert scholar == 1.0


def test_a_game_that_runs_out_of_moves_goes_to_the_side_well_ahead():
    assert lead(board_after()) == 0 and lead(board_after("e4", "d5", "exd5")) == 1
    # White wins a queen for nothing, and the game stops there.
    up = play(scripted("e4", "exd5", "Nc3", "Nxd5"), scripted("d5", "Qxd5", "Qd6", "a6"), most=7)
    assert up == 0.5  # after 7 moves white is a pawn up: not enough
    won = play(
        scripted("e4", "exd5", "Nc3", "Nb5", "Nxd6+"), scripted("d5", "Qxd5", "Qd6", "a6"), most=9
    )
    assert won == 1.0  # a queen and a pawn up


def test_the_grabber_mates_or_takes_the_best_piece_on_offer():
    generator = torch.Generator().manual_seed(0)
    player = grabber(generator)
    assert player(board_after("f3", "e5", "g4"), []) == "Qh4#"  # a mate beats everything
    queen_or_pawn = board_after("e4", "d5", "Nc3", "Qd6", "Nb5", "a6")  # Nxd6+ takes a queen
    assert player(queen_or_pawn, []) == "Nxd6+"
    assert player(board_after(), []) in {board_after().san(m) for m in board_after().legal_moves}


def test_the_random_mover_spreads_over_the_legal_moves():
    player = random_mover(torch.Generator().manual_seed(0))
    seen = {player(board_after(), []) for _ in range(300)}
    assert len(seen) == 20  # each of white's 20 first moves, sooner or later


def test_a_match_swaps_colours_and_a_stronger_player_scores_more():
    generator = torch.Generator().manual_seed(0)
    scores = match(grabber(generator), random_mover(generator), 30)
    assert len(scores) == 30 and sum(scores) / 30 > 0.6


def test_an_untrained_model_still_finishes_a_legal_game():
    words = Tokenizer(train(" e4 e5 Nf3 Nc6 Bb5 a6" * 40, 300))
    torch.manual_seed(0)
    model = GPT(vocab=words.end + 1, context=256, width=32, heads=4, layers=2)
    generator = torch.Generator().manual_seed(0)
    player, counts = model_mover(model, words, generator, tries=2)
    score = play(player, random_mover(generator), most=12)  # each move it is asked for is legal
    assert score in (0.0, 0.5, 1.0) and counts["moves"] == 6
    assert counts["slips"] >= 5  # it knows no chess: the rules pick nearly each move for it


def test_the_elo_rule_by_hand():
    assert expected(1500, 1500) == 0.5
    assert expected(1900, 1500) == pytest.approx(10 / 11)  # 400 points: ten to one
    assert expected(1500, 1900) == pytest.approx(1 / 11)


def test_ratings_are_recovered_from_games_played_by_the_rule():
    truth = {"a": 0.0, "b": 200.0, "c": 500.0}
    generator = torch.Generator().manual_seed(0)
    results = []
    for one, other in [("a", "b"), ("a", "c"), ("b", "c")]:
        wins = torch.rand(2000, generator=generator) < expected(truth[one], truth[other])
        results += [(one, other, float(win)) for win in wins]
    fitted = rate(results, ["a", "b", "c"])
    assert fitted["a"] == 0
    assert fitted["b"] == pytest.approx(200, abs=25) and fitted["c"] == pytest.approx(500, abs=35)
