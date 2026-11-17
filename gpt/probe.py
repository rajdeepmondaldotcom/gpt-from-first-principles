"""Lesson 40. The board in its head: read a chess position out of the model's inner numbers."""

import chess  # the rules of chess, from the `chess` extra
import torch
from torch import nn

from gpt.optim import AdamW
from gpt.tensors import cross_entropy

PIECES = 13  # empty, six kinds of white piece, six kinds of black piece


@torch.no_grad()
def streams(model, ids, edit=None):
    """The stream after each block, for one text. Returns a list: one (tokens, width) grid a block.

    `edit` is (block, change): the change is added to that block's result at the last token,
    and each later block reads the changed stream.
    """
    x, out = model.embed(torch.tensor([ids])), []
    for at, block in enumerate(model.blocks):
        x = block(x)
        if edit and edit[0] == at:
            x[0, -1] += edit[1]
        out.append(x[0])
    return out


@torch.no_grad()
def lens(model, stream):
    """The scores the model would give if this stream were its last: the final norm, then the
    layer that turns a stream into scores."""
    return model.out(model.norm(stream))


def squares(played):
    """The board after these moves, as 64 numbers from a1 to h8: 0 for empty, 1 to 6 for a white
    pawn, knight, bishop, rook, queen and king, 7 to 12 for black's."""
    board = chess.Board()
    for move in played:
        board.push_san(move)
    out = [0] * 64
    for square, piece in board.piece_map().items():
        out[square] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return out


def ends(words, played):
    """The place of each move's last token, in a text that starts with the end token."""
    places, at = [], 0
    for move in played:
        at += len(words.encode(" " + move))
        places.append(at)
    return places


def fit(x, y, generator, steps=1000, rate=1e-2):
    """A linear probe: one layer from a stream to 13 scores for each of 64 squares, trained to
    name the piece on each square. x: (positions, width). y: (positions, 64)."""
    probe = nn.Linear(x.size(1), 64 * PIECES)
    optimizer = AdamW(probe.parameters(), rate=rate, decay=0.0)
    for _ in range(steps):
        picks = torch.randint(len(x), (256,), generator=generator)
        loss = cross_entropy(probe(x[picks]).view(-1, PIECES), y[picks].view(-1))
        probe.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


@torch.no_grad()
def read(probe, x):
    """The piece the probe names on each square. Returns (positions, 64)."""
    return probe(x).view(len(x), 64, PIECES).argmax(dim=-1)


def erase(probe, stream, square, piece, margin=3.0):
    """The smallest change to a stream, along the probe's own line, after which the probe scores
    "empty" on `square` above `piece`, by `margin`."""
    rows = probe.weight.view(64, PIECES, -1)[square]
    biases = probe.bias.view(64, PIECES)[square]
    line = rows[0] - rows[piece]  # toward "empty", away from the piece
    lead = (stream @ line + biases[0] - biases[piece]).item()  # how far "empty" leads now
    return (margin - lead) / (line @ line).item() * line.detach()


if __name__ == "__main__":
    from pathlib import Path

    from gpt.data import documents, read
    from gpt.evaluate import interval
    from gpt.results import save, say
    from gpt.story import load
    from gpt.tensors import softmax

    cache = Path("data/cache")
    model, words = load(cache / "chess")
    generator = torch.Generator().manual_seed(0)
    unseen = (cache / "chess_val.txt").read_text().split("\n")[:300]
    seen = documents(read(cache / "chess_train.bin")[:400_000], words.end)
    seen = [words.decode(doc[1:-1].tolist()).strip() for doc in seen][:1500]

    def positions(games, every=4, most=100):
        """Each `every`-th position of each game: the stream after each block where the move
        ends, the board there, the moves so far, and which game it came from."""
        xs, boards, lines, game_of = [[] for _ in model.blocks], [], [], []
        for number, game in enumerate(games):
            played = game.split()[:most]
            seen_streams = streams(model, [words.end, *words.encode(" " + " ".join(played))])
            places = ends(words, played)
            for j in range(every - 1, len(played), every):
                for block, stream in enumerate(seen_streams):
                    xs[block].append(stream[places[j]])
                boards.append(squares(played[: j + 1]))
                lines.append(played[: j + 1])
                game_of.append(number)
        return [torch.stack(x) for x in xs], torch.tensor(boards), lines, game_of

    x, y, _, _ = positions(seen)
    held_x, held_y, held_lines, game_of = positions(unseen)
    say(f"positions to fit on: {len(y):,} from 1,500 games. Held back: {len(held_y):,} from 300")

    def by_game(right):
        """A share of squares named right, with a 95% interval drawn game by game."""
        blocks = [[] for _ in range(300)]
        for number, row in zip(game_of, right.tolist()):
            blocks[number] += row
        blocks = [block for block in blocks if block]
        low, high = interval(blocks, generator)
        return right.float().mean().item(), low, high

    usual = torch.stack([y[:, s].bincount(minlength=PIECES).argmax() for s in range(64)])
    share, low, high = by_game(held_y == usual)
    say(f"naming each square's most usual piece: {share:.1%} right, {low:.1%} to {high:.1%}")
    probes, accuracy = [], {}
    for block in range(len(model.blocks)):
        probes.append(fit(x[block], y, generator))
        share, low, high = by_game(read(probes[-1], held_x[block]) == held_y)
        accuracy[block + 1] = share
        say(
            f"a probe on the stream after block {block + 1}: {share:.1%} right, {low:.1%} to {high:.1%}"
        )
    best = max(accuracy, key=accuracy.get) - 1
    mixed = y[torch.randperm(len(y), generator=generator)]  # boards given to the wrong streams
    control = fit(x[best], mixed, generator)
    share, low, high = by_game(read(control, held_x[best]) == held_y)
    say(f"the same probe fitted to shuffled boards: {share:.1%} right, {low:.1%} to {high:.1%}")
    moved = held_y != usual  # squares that are not as they usually are: the hard ones
    hard = (read(probes[best], held_x[best]) == held_y)[moved].float().mean().item()
    say(f"on squares that differ from their usual piece, the probe is right {hard:.1%} of the time")

    # Rub the mover's queen out of the stream after that block. Does it play its queen less?
    queen_moves = [i for i, token in words.bytes.items() if token.startswith(b" Q")]

    def queen_chance(ids, edit=None):
        """The chance the model gives to a queen move next, with or without an edit."""
        last = streams(model, ids, edit)[-1][-1]
        return softmax(lens(model, last))[queen_moves].sum().item()

    plain, no_queen, no_pawn, game = [], [], [], []
    for line, number in zip(held_lines, game_of):
        board = chess.Board()
        for move in line:
            board.push_san(move)
        queens = list(board.pieces(chess.QUEEN, board.turn))
        pawns = list(board.pieces(chess.PAWN, board.turn))
        can_move = len(queens) == 1 and any(m.from_square == queens[0] for m in board.legal_moves)
        if not (can_move and pawns):
            continue  # we want one queen that has a move, and a pawn to rub out for comparison
        ids = [words.end, *words.encode(" " + " ".join(line))]
        stream = streams(model, ids)[best][-1]
        colour = 0 if board.turn == chess.WHITE else 6
        plain.append(queen_chance(ids))
        for square, piece, out in (
            (queens[0], 5 + colour, no_queen),
            (pawns[0], 1 + colour, no_pawn),
        ):
            out.append(queen_chance(ids, (best, erase(probes[best], stream, square, piece))))
        game.append(number)

    def mean_and_range(values, base):
        """The mean of `values`, and a 95% interval for its change from `base`, game by game."""
        blocks = {}
        for number, value, start in zip(game, values, base):
            blocks.setdefault(number, []).append(value - start)
        low, high = interval(list(blocks.values()), generator)
        return sum(values) / len(values), low, high

    say(f"positions where the mover has one queen that can move: {len(plain):,}")
    say(f"chance of a queen move next, untouched: {sum(plain) / len(plain):.1%}")
    edits = {}
    for name, values in (("the queen", no_queen), ("a pawn", no_pawn)):
        mean, low, high = mean_and_range(values, plain)
        edits[name] = mean
        say(
            f"with {name} rubbed out after block {best + 1}: {mean:.1%}, "
            f"a change of {low:+.1%} to {high:+.1%}"
        )

    # The move forming: read the stream after each block through the final layer.
    final = lens(model, held_x[-1]).argmax(dim=-1)  # the token on top at the end, for each position
    formed = {}
    for block in range(len(model.blocks)):
        chances = softmax(lens(model, held_x[block]))
        on_top = (chances.argmax(dim=-1) == final).float().mean().item()
        given = chances[torch.arange(len(final)), final].mean().item()
        formed[block + 1] = on_top
        say(
            f"after block {block + 1}: the final choice is on top in {on_top:.0%} of positions, "
            f"with a chance of {given:.0%}"
        )
    save(
        "40_probe",
        torch=torch.__version__,
        device="cpu",
        positions={"fit": len(y), "held": len(held_y), "edited": len(plain)},
        accuracy=accuracy,
        control=share,
        hard_squares=hard,
        queen_chance={"plain": sum(plain) / len(plain), **edits},
        formed=formed,
    )
