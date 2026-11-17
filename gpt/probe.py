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
