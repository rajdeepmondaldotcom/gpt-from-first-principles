"""Lesson 39. Can it play? Whole games against simple players, and a rating with a range."""

import chess  # the rules of chess, from the `chess` extra
import torch

from gpt.games import next_move
from gpt.optim import AdamW

WORTH = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def random_mover(generator):
    """A player that picks any legal move, each with the same chance."""

    def move(board, played):
        legal = [board.san(m) for m in board.legal_moves]
        return legal[int(torch.randint(len(legal), (1,), generator=generator))]

    return move


def grabber(generator):
    """A player that looks one move ahead: it mates if it can, or else takes the most valuable
    piece on offer, or else moves at random."""

    def move(board, played):
        best, choices = -1, []
        for m in board.legal_moves:
            worth = WORTH.get(board.piece_type_at(m.to_square), 0)
            board.push(m)
            worth = 100 if board.is_checkmate() else worth
            board.pop()
            if worth > best:
                best, choices = worth, []
            if worth == best:
                choices.append(board.san(m))
        return choices[int(torch.randint(len(choices), (1,), generator=generator))]

    return move


def model_mover(model, words, generator, tries=5):
    """The model as a player. It writes its top move. If that is not legal it draws `tries - 1`
    more by chance. If none is legal the rules pick a move for it, and `slips` counts that.

    Returns the player and its counts.
    """
    counts = {"moves": 0, "slips": 0}

    def move(board, played):
        counts["moves"] += 1
        legal = {board.san(m) for m in board.legal_moves}
        for attempt in range(tries):
            written = next_move(
                model, words, played, temperature=min(attempt, 1), generator=generator
            )
            try:
                return board.san(board.parse_san(written))  # the move, as the rules spell it
            except ValueError:
                pass
        counts["slips"] += 1
        return sorted(legal)[int(torch.randint(len(legal), (1,), generator=generator))]

    return move, counts


def lead(board):
    """White's pieces minus black's, in pawns: 3 for a knight or bishop, 5 a rook, 9 a queen."""
    return sum(
        WORTH.get(piece.piece_type, 0) * (1 if piece.color == chess.WHITE else -1)
        for piece in board.piece_map().values()
    )


def play(white, black, most=100):
    """One game. Returns white's score: 1 for a win, 0 for a loss, 0.5 for a draw.

    After `most` moves, counting both sides, the game stops, and a side 3 pawns or more ahead
    takes it. The model reads 256 tokens, and 100 moves are about 210.
    """
    board, played = chess.Board(), []
    while not board.is_game_over() and len(played) < most:
        move = (white if board.turn == chess.WHITE else black)(board, played)
        board.push_san(move)
        played.append(move)
    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0  # the side to move has been mated
    ahead = 0 if board.is_game_over() else lead(board)
    return 1.0 if ahead >= 3 else 0.0 if ahead <= -3 else 0.5


def match(first, second, games):
    """`games` games, with the colours swapped each time. Returns the first player's scores."""
    return [play(first, second) if i % 2 == 0 else 1 - play(second, first) for i in range(games)]


def expected(mine, theirs):
    """The score the Elo rule expects for a player rated `mine` against one rated `theirs`."""
    return 1 / (1 + 10 ** ((theirs - mine) / 400))


def rate(results, players, steps=400):
    """One rating for each player, from results: (player, other player, the first one's score).

    The ratings are the ones that make the results most likely. The first player is held at 0.
    """
    a = torch.tensor([players.index(one) for one, _, _ in results])
    b = torch.tensor([players.index(other) for _, other, _ in results])
    score = torch.tensor([s for _, _, s in results])
    free = torch.zeros(len(players) - 1, requires_grad=True)
    optimizer = AdamW([free], rate=0.05)
    for _ in range(steps):
        ratings = torch.cat([torch.zeros(1), free])  # in units of 400 points
        guess = expected(400 * ratings[a], 400 * ratings[b]).clamp(1e-6, 1 - 1e-6)
        loss = -(score * guess.log() + (1 - score) * (1 - guess).log()).mean()
        free.grad = None
        loss.backward()
        optimizer.step()
    return dict(zip(players, (400 * torch.cat([torch.zeros(1), free.detach()])).tolist()))


if __name__ == "__main__":
    from itertools import combinations
    from pathlib import Path

    from gpt.evaluate import interval, rate_interval
    from gpt.results import save, say
    from gpt.story import load

    cache = Path("data/cache")
    generator = torch.Generator().manual_seed(0)
    model, words = load(cache / "chess")
    younger, _ = load(cache / "chess_early")  # the same run, stopped at step 2,500
    full, slips = model_mover(model, words, generator)
    early, early_slips = model_mover(younger, words, generator)
    players = {
        "the random mover": random_mover(generator),
        "the grabber": grabber(generator),
        "the model at step 2,500": early,
        "the model": full,
    }

    results, table = [], {}
    for (weaker, first), (stronger, second) in combinations(players.items(), 2):
        scores = match(second, first, 60)
        whole = sum(scores) in (0, 60)  # one side took each point: the bootstrap has no width
        low, high = (
            rate_interval(scores, generator)
            if whole
            else interval([[s] for s in scores], generator)
        )
        table[f"{stronger} against {weaker}"] = sum(scores) / 60
        say(
            f"{stronger} against {weaker}: {sum(scores) / 60:.2f} of the points in 60 games, "
            f"95% interval {low:.2f} to {high:.2f}"
        )
        results += [(stronger, weaker, s) for s in scores]
    for name, counts in (("the model", slips), ("the model at step 2,500", early_slips)):
        share = counts["slips"] / counts["moves"]
        say(
            f"{name}: the rules picked {counts['slips']:,} of its {counts['moves']:,} moves, {share:.1%}"
        )

    # One rating each. A made-up draw between each pair keeps a rating from running off when one
    # player takes each point from another.
    names = list(players)
    ties = [(a, b, 0.5) for a, b in combinations(names, 2)]
    ratings = rate(results + ties, names)
    again = []
    for _ in range(100):  # the bootstrap of lesson 24, over games
        picks = torch.randint(len(results), (len(results),), generator=generator).tolist()
        again.append(rate([results[i] for i in picks] + ties, names, steps=200))
    ranges = {}
    for name in names[1:]:
        fitted = torch.tensor([one[name] for one in again])
        ranges[name] = [fitted.quantile(0.025).item(), fitted.quantile(0.975).item()]
        say(
            f"{name}: rated {ratings[name]:+.0f} with the random mover at 0, "
            f"95% interval {ranges[name][0]:+.0f} to {ranges[name][1]:+.0f}"
        )
    save(
        "39_play",
        torch=torch.__version__,
        device="cpu",
        points=table,
        slips={"model": slips, "early": early_slips},
        ratings=ratings,
        ranges=ranges,
    )
