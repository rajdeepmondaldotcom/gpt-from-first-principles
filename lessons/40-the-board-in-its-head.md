# 40. The board in its head

By the end of this page you have read a chess position out of the model's inner numbers, and changed the model's move by rubbing one piece out of them.

The model plays chess from a list of moves. We never showed it a board. A legal move depends on where each piece stands, and we have not looked for the pieces inside the model.

## The idea

After the model reads a move's last token, each block leaves a row of 256 numbers there: lesson 9's stream. The next move is worked out from those numbers, and they carry no labels. We look for the position in them.

A probe is the plainest reader we can build: one linear layer, 256 numbers in, and 13 scores out for each of the 64 squares, one for empty and one for each kind of piece of each colour. We train it on each fourth position of a game, so white is always to move, and a chess program gives the right answer by replaying the moves. We freeze the model and only read it.

We want it weak: a big network could work the position out for itself. One linear layer cannot, so if it reads the board, the stream holds the board along straight lines.

We check three things.

1. Test on games the probe has not seen, and leave out each position whose moves so far also begin one of its own games: openings repeat.
2. Compare with a lazy guess: each square's most usual piece. Score the probe apart on the squares where that guess is wrong.
3. Shuffle the answers. Fit the same probe to the same streams, with each board handed to another position's stream. It can still learn what is usual, and nothing about this game.

The board could be in the stream and play no part in the move. So we change it, after whichever of the first five blocks reads best on other games: a block still has to run after the edit.

Take a position where white has a pawn and one queen that can move. Along the probe's line for "queen on this square", nudge the stream until the probe scores empty above queen by a set margin, and let the model run on. A model that works from this board should give queen moves less chance. A nudge of the same length along a pawn's line should not.

Last, we watch the move form. Lesson 10's last two steps, a normalization and the scores layer, show what the model would say if it stopped after an earlier block.

## By hand

A board as 64 numbers, from a1 to h8. An empty square is 0, a white pawn, knight, bishop, rook, queen and king are 1 to 6, and black's are 7 to 12.

| After | Square e2 | Square e4 | Square f3 | Square g1 |
|---|---|---|---|---|
| no moves | 1 | 0 | 0 | 2 |
| `e4` | 0 | 1 | 0 | 2 |
| `e4 e5 Nf3` | 0 | 1 | 2 | 0 |

The size of a probe. It maps 256 numbers to 64 × 13 = 832 scores: 212,992 weights and 832 biases.

The nudge. Say the probe's score for "empty" on a square trails its score for "queen" by 4, and we want it to lead by 3. Adding the line between the two raises the lead by the line's squared length. If that is 2, we add (3 + 4) / 2 = 3.5 times the line.

## The code

```python
# gpt/probe.py
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
```

An ordinary training loop on one layer, 256 positions to a step. `view(-1, PIECES)` lays the scores out as a row of 13 for each square, so that lesson 3's loss can mark each square.

```python
# gpt/probe.py
def erase(probe, stream, square, piece, margin=3.0, length=None):
    """The smallest change to a stream, along the probe's own line, after which the probe scores
    "empty" on `square` above `piece`, by `margin`. No change where it does so already.

    With `length`, a change of that length along the same line, whatever the probe reads now.
    """
    rows = probe.weight.view(64, PIECES, -1)[square]
    biases = probe.bias.view(64, PIECES)[square]
    line = rows[0] - rows[piece]  # toward "empty", away from the piece
    if length is not None:
        return length / line.norm().item() * line.detach()
    lead = (stream @ line + biases[0] - biases[piece]).item()  # how far "empty" leads now
    return max(margin - lead, 0.0) / (line @ line).item() * line.detach()
```

`view(64, PIECES, -1)` sorts the layer's rows by square and piece. The row for "empty" minus the row for the piece is the one line in the stream along which the probe tells the two apart.

## The proof

```bash
uv run pytest tests/test_probe.py
```

- The hand table for the start and after `e4 e5 Nf3`, and the place of each move's last token.
- The last block's stream, put through the final layer, gives the model's own scores. An edit after one block changes the last token's stream from that block on, and no earlier token's.
- On made-up streams with a board written in along straight lines, a probe reads over 0.9 of squares, and under 0.4 when fitted to shuffled boards.
- After `erase`, the probe scores empty above the piece by the margin asked for. Where it did already, nothing changes. A nudge of a set length runs along the same line.

```bash
uv run python -m gpt.probe
```

```
positions to fit on: 18,940 from 1,200 games. To pick a block: 4,473 from 300. Held back: 4,608 from 300
naming each square's most usual piece: 67.3% right, 66.7% to 67.8%
a probe on the stream after block 1: 73.8% right, 73.2% to 74.3%
a probe on the stream after block 2: 75.6% right, 75.1% to 76.2%
a probe on the stream after block 3: 76.2% right, 75.7% to 76.7%
a probe on the stream after block 4: 78.7% right, 78.2% to 79.1%
a probe on the stream after block 5: 80.7% right, 80.3% to 81.2%
a probe on the stream after block 6: 81.2% right, 80.7% to 81.6%
the probe after block 5, fitted to shuffled boards: 63.2% right, 62.6% to 63.7%
on squares that differ from their usual piece, that probe is right 60.9% of the time
positions where the mover has one queen that can move: 3,200
chance of a queen move next, untouched: 20.3%
a nudge 0.07 of the stream's length after block 5, rubbing out the queen: 20.1%, a change of -0.3% to -0.1%
a nudge 0.07 of the stream's length after block 5, rubbing out the pawn: 20.3%, a change of -0.0% to +0.0%
a nudge 0.12 of the stream's length after block 5, rubbing out the queen: 19.9%, a change of -0.5% to -0.3%
a nudge 0.12 of the stream's length after block 5, rubbing out the pawn: 20.3%, a change of -0.0% to +0.1%
a nudge 0.28 of the stream's length after block 5, rubbing out the queen: 18.5%, a change of -2.0% to -1.5%
a nudge 0.28 of the stream's length after block 5, rubbing out the pawn: 20.5%, a change of +0.1% to +0.3%
after block 1: the final choice is on top in 5% of positions, with a chance of 3%
after block 2: the final choice is on top in 14% of positions, with a chance of 10%
after block 3: the final choice is on top in 19% of positions, with a chance of 13%
after block 4: the final choice is on top in 27% of positions, with a chance of 18%
after block 5: the final choice is on top in 43% of positions, with a chance of 28%
after block 6: the final choice is on top in 100% of positions, with a chance of 35%
a probe after block 5 on positions with either side to move: 74.9% right, 74.3% to 75.3%
```

Guessing each square's usual piece is right for 67.3% of squares. The probe after block 1 reads 73.8%, rising block by block to 81.2%. Fitted to shuffled boards, the probe after block 5 reads 63.2%. Fitted to the true boards it reads 80.7%, and 60.9% on squares that are not as usual.

Untouched, the model gives queen moves a chance of 20.3%. A margin of 3 takes a nudge of 0.07 of the stream's length on average, and brings that to 20.1%. Margins of 10 and 30, at 0.12 and 0.28, bring it to 19.9% and 18.5%, each interval clear of zero. Pawn nudges of the same lengths leave 20.3%, 20.3% and 20.5%, the last 0.1 to 0.3 points up. So the model uses the probe's line: the longest nudge costs queen moves 1.5 to 2.0 points.

The final choice is on top in 27% of positions after block 4, and 43% after block 5.

A last probe is fitted and tested on each third position, where either side may be to move. After block 5 it reads 74.9%, against 80.7% with white to move.

## What it still cannot do

Every run so far used one laptop: a storyteller of 5,374,464 parameters and a chess player of 4,488,448, from one file of model code. We have not run that code on hardware built for it.

## Check yourself

Predict before you run: a probe fitted to shuffled boards has never been told what is on any board in front of it. How many squares will it still name right?

<details><summary>Answer</summary>

63.2%, close to the usual piece's 67.3%. Most squares hold what they usually hold, and a probe with shuffled answers can still learn that.
</details>

<details><summary>From lesson 9: why is the stream after a block the right place to look?</summary>

Each block reads the stream and adds to it, and the final layer reads it to choose the move. Anything one block hands to a later one passes through it.
</details>

Go deeper:

- Karvonen, [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (2024). Linear probes and edits on a chess model.
- Nanda, Lee and Wattenberg, [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941) (2023). On Othello, one linear layer reads the board once squares are labelled as the mover's and the opponent's: the thing to try on our 74.9%.
