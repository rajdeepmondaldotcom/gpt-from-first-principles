import pytest
import torch

import gpt.model
from gpt.bpe import Tokenizer, train
from gpt.games import games
from gpt.model import GPT

pa = pytest.importorskip("pyarrow")  # the chess extra: Lichess publishes its games in this format
from gpt.strong import MOVES, best_legal, fused, rating, strong


def test_only_games_between_two_strong_players_are_kept(tmp_path):
    rows = [  # the kind of game, white's rating, black's rating, the move that tells them apart
        ("Rated Blitz game", 2300, 2250, "4. Ba4"),
        ("Rated Blitz game", 2300, 2199, "4. Bxc6"),  # one player too weak
        ("Rated Blitz game", 2199, 2300, "4. Bc4"),  # the other one
        ("Rated Blitz game", None, 2400, "4. Be2"),  # a rating not known
        ("Rated Bullet game", 2500, 2500, "4. Bf1"),  # too fast to learn from
        ("Rated Rapid tournament https://lichess.org/tournament/x", 2200, 2200, "4. Bd3"),
    ]
    table = pa.table(
        {
            "Event": [row[0] for row in rows],
            "WhiteElo": pa.array([row[1] for row in rows], pa.int16()),
            "BlackElo": pa.array([row[2] for row in rows], pa.int16()),
            "movetext": [
                f"1. e4 {{ [%clk 0:05:00] }} 1... e5 2. Nf3 Nc6 3. Bb5 a6 {row[3]} 1-0"
                for row in rows
            ],
        }
    )
    pa.parquet.write_table(table, tmp_path / "games.parquet")
    kept = list(games(strong(tmp_path / "games.parquet", 2200), least=1))
    assert kept == ["e4 e5 Nf3 Nc6 Bb5 a6 Ba4", "e4 e5 Nf3 Nc6 Bb5 a6 Bd3"]
    assert len(strong(tmp_path / "games.parquet", 2250)) == 1  # a higher bar keeps only the first


def test_a_common_move_is_one_token_and_a_game_comes_back_as_it_went_in():
    text = " e4 e5 Nf3 Nc6 Bb5 a6" * 20 + " O-O-O+ exd8=Q#"
    words = Tokenizer(train(text, 300, MOVES), MOVES)
    assert len(words.encode(" Nf3")) == 1 and len(words.encode(" e4 e5 Nf3")) == 3
    assert words.decode(words.encode(text)) == text
    # Lesson 5's cut parts the letters from the digit, so there a move is two tokens or more.
    assert len(Tokenizer(train(text, 300)).encode(" Nf3")) == 2


def test_the_fused_attention_gives_the_models_own_scores(monkeypatch):
    torch.manual_seed(0)
    model = GPT(vocab=64, context=32, width=48, heads=4, kv_heads=2, layers=2)
    ids = torch.randint(64, (2, 32))
    ours = model(ids)
    monkeypatch.setattr(gpt.model, "attend", fused)
    assert torch.allclose(model(ids), ours, atol=1e-5)
    kept = [{} for _ in model.blocks]  # and with a cache: 20 tokens, then the rest one at a time
    parts = [model(ids[:, :20], kept)] + [model(ids[:, i : i + 1], kept) for i in range(20, 32)]
    assert torch.allclose(torch.cat(parts, dim=1), ours, atol=1e-5)


def test_the_rating_is_the_one_that_makes_the_results_most_likely():
    # Lesson 39's rule: 400 points above an opponent takes 10 points for each 1 it gives.
    assert abs(rating([(1500, 1.0)] * 100 + [(1500, 0.0)] * 10) - 1900) <= 1
    assert rating([(1500, 0.5)] * 40) == 1500
    # Two opponents, each scored as a player rated 1,700 would score: 0.76 and 0.24.
    mixed = [(1500, 1.0)] * 76 + [(1500, 0.0)] * 24 + [(1900, 1.0)] * 24 + [(1900, 0.0)] * 76
    assert abs(rating(mixed) - 1700) <= 2


def test_the_player_makes_the_legal_move_the_model_scores_highest():
    chess = pytest.importorskip("chess")
    text = " e4 e5 Nf3 Nc6 d4 d5 Nc3 Nf6 a3 a6 h3 h6" * 5
    words = Tokenizer(train(text, 320, MOVES), MOVES)
    torch.manual_seed(0)
    model = GPT(vocab=words.end + 1, context=32, width=48, heads=4, kv_heads=2, layers=2)
    board = chess.Board()
    played = best_legal(model, words)(board, [])
    scores = model(torch.tensor([[words.end]]))[0, -1]
    first = {
        san: scores[words.vocab[f" {san}".encode()]]
        for san in ("e4", "Nf3", "d4", "Nc3", "a3", "h3")
    }
    assert played == max(first, key=first.get)  # of white's first moves that are one token
    board.push_san("e4")
    assert best_legal(model, words)(board, ["e4"]) in ("e5", "Nc6", "d5", "Nf6", "a6", "h6")
    # A position where no legal move is a token the tokenizer knows: it still plays a legal move.
    lone = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
    assert lone.parse_san(best_legal(model, words)(lone, [])) in lone.legal_moves
