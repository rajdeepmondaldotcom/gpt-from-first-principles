import pytest
import torch

from gpt.bpe import Tokenizer, train
from gpt.games import cuts, games, legal, legal_moves, moves, next_move, opens_a_game
from gpt.gpt2 import loss_on
from gpt.model import GPT
from gpt.optim import AdamW

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
    assert not legal(["e4", "e5"], "e5")  # black's pawn stands there
    assert legal(["e4", "e5"], "Qh5") and not legal([], "Qh5")  # open only once the pawn has moved
    assert not legal([], "hello") and not legal([], "")  # not a move at all
    assert legal(["f3", "e5", "g4"], "Qh4#") and not legal(["f3", "e5", "g4", "Qh4#"], "a3")


LINE = "e4 e5 Nf3 Nc6 Bb5 a6 O-O"


def learned_one_game():
    """A tokenizer and a small model that has learned one game by heart."""
    words = Tokenizer(train((" " + LINE) * 40, 300))
    ids = torch.tensor([[words.end, *words.encode(" " + LINE), words.end]])
    torch.manual_seed(0)
    model = GPT(vocab=words.end + 1, context=64, width=32, heads=4, layers=2)
    optimizer = AdamW(model.parameters(), rate=1e-2)
    for _ in range(150):
        model.zero_grad()
        loss_on(model, ids[:, :-1], ids[:, 1:]).backward()
        optimizer.step()
    return model, words


def test_a_model_that_knows_one_game_writes_its_next_move():
    model, words = learned_one_game()
    played = LINE.split()
    assert next_move(model, words, []) == "e4"  # from the end token alone: the first move
    assert next_move(model, words, played[:2]) == "Nf3"  # the next move, and only that one
    assert next_move(model, words, played[:6]) == "O-O"  # three tokens, read as one move
    assert legal_moves(model, words, [played[:2], played[:6], []]) == [True, True, True]


def test_a_pass_is_not_a_move():
    assert not legal([], "--") and not legal(["e4"], "0000")  # the library reads both as a pass


def test_each_game_is_cut_inside_its_first_moves():
    generator = torch.Generator().manual_seed(0)
    game = " ".join(["Nf3", "Nf6", "Ng1", "Ng8"] * 30)  # 120 moves
    lengths = [len(cut) for cut in cuts([game] * 300, generator)]
    assert max(lengths) == 59 and min(lengths) == 0  # among the first 60 moves, never past them
    assert all(len(cut) < 7 for cut in cuts([LINE] * 50, generator))  # a 7-move game: inside it


def test_a_position_is_found_among_the_games_it_opens():
    one, two = "e4 e5 Nf3 Nc6", "d4 d5 c4 e6"
    words = Tokenizer(train(" " + one + " " + two, 280))
    ids = torch.tensor([i for game in (one, two) for i in (*words.encode(" " + game), words.end)])
    starts = torch.tensor([0, len(words.encode(" " + one)) + 1])
    assert opens_a_game(words, ids, starts, []) and opens_a_game(words, ids, starts, ["e4", "e5"])
    assert opens_a_game(words, ids, starts, two.split()[:3])
    assert not opens_a_game(words, ids, starts, ["e4", "d5"])  # no game goes that way
    assert not opens_a_game(words, ids, starts, ["d4", "d5", "c"])  # half a move opens nothing
