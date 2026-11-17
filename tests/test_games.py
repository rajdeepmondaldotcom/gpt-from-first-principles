import pytest
import torch

from gpt.bpe import Tokenizer, train
from gpt.games import games, legal, legal_moves, moves, next_move
from gpt.model import GPT

chess = pytest.importorskip("chess")  # these tests need the chess extra

RECORD = """[Event "Rated Blitz game"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Bullet game"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1

[Event "Rated Classical game"]
[Result "1/2-1/2"]

1. d4 { [%eval 0.2] } 1... d5?! { [%eval 0.4] } 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 8. cxd5 exd5 9. Bd3 Bb7 10. O-O Nbd7 1/2-1/2
"""


def test_a_line_of_a_record_becomes_moves_alone():
    assert moves("1. e4 e5 2. Nf3 Nc6 1-0") == "e4 e5 Nf3 Nc6"
    assert moves("1. d4 { [%eval 0.2] } 1... d5?! 2. c4 e6 1/2-1/2") == "d4 d5 c4 e6"
    assert moves("1. e4 c5 2. O-O-O exd8=Q+ 3. Qh4# *") == "e4 c5 O-O-O exd8=Q+ Qh4#"


def test_short_games_are_left_out_and_the_rest_are_legal_from_the_first_move():
    kept = list(games(RECORD.splitlines()))
    assert len(kept) == 2 and kept[0].startswith("e4 e5 Nf3") and kept[1].startswith("d4 d5 c4")
    for game in kept:  # each move is legal after the ones before it
        played = game.split()
        assert all(legal(played[:i], played[i]) for i in range(len(played)))


def test_legal_by_hand():
    assert legal([], "e4") and legal(["e4", "e5"], "Nf3")
    assert not legal([], "e5")  # a white pawn cannot reach e5 in one move
    assert not legal(["e4"], "e4")  # it is black's turn, and e4 is taken
    assert not legal(["e4", "e5"], "Ke3")  # a king moves one square
    assert not legal([], "hello") and not legal([], "")  # not a move at all
    assert legal(["f3", "e5", "g4"], "Qh4#") and not legal(["f3", "e5", "g4", "Qh4#"], "a3")


def test_the_models_move_is_its_text_up_to_the_next_space():
    text = " e4 e5 Nf3 Nc6 Bb5 a6" * 40
    words = Tokenizer(train(text, 300))
    torch.manual_seed(0)
    model = GPT(vocab=words.end + 1, context=64, width=32, heads=4, layers=2)
    move = next_move(model, words, ["e4", "e5"])  # an untrained model: any text, or none
    assert " " not in move
    generator = torch.Generator().manual_seed(0)
    hits = legal_moves(model, words, ["e4 e5 Nf3 Nc6 Bb5 a6"] * 5, generator)
    assert len(hits) == 5 and all(hit in (True, False) for hit in hits)
