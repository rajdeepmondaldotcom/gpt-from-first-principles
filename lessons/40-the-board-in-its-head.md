# 40. The board in its head

By the end of this page you have read a chess position out of the model's inner numbers, square by square, and changed the model's move by rubbing one piece out of them.

The model plays chess from a list of moves. It was never shown a board. To move legally it has to know where the pieces are, and we have not looked for that knowledge inside it.

## The idea

After the model reads the last token of a move, each block leaves a row of 256 numbers at that place: the stream of lesson 9. Whatever the model knows about the game at that point is in those numbers, because the next move is worked out from them. They do not come with labels. The question is whether the position can be read from them.

A probe is the plainest reader we can build. It is one linear layer and nothing else: 256 numbers in, and for each of the 64 squares, 13 scores out, one for empty and one for each kind of piece of each colour. We train it on positions from games, where the right answer is known because a chess program can replay the moves. The model itself is frozen and is only read.

A probe this weak is the point. A big network on top of the stream could work the position out for itself from scraps. One linear layer cannot work much out. If it reads the board, the board was there to be read, laid out along straight lines in the stream.

Three checks keep us honest.

1. Test on games the probe has not seen. A probe can store its training positions as a model can store its text.
2. Compare with a lazy guess. Most squares hold what they usually hold: a rook sits on a1 for much of most games. So we count how often "the most usual piece for this square" is right, and score the probe separately on the squares where that guess is wrong.
3. Shuffle the answers. Fit the same probe to the same streams, with each board handed to another position's stream. It can still learn what is usual, and nothing about this game. The gap between the two probes is what the stream holds about the position in front of it.

Reading is not using. The board could be written in the stream and play no part in choosing the move. So we change it. Take a position where the side to move has its queen. We fit a probe after each block, and work after the block whose probe reads best. Along that probe's own line for "queen on this square", nudge the stream until the probe scores empty well above queen there, and let the rest of the model run on. If the model works from this board, it should now give less chance to queen moves. A nudge of the same kind that rubs out a pawn should leave them about as they were.

Last, we watch the move form. Lesson 10's last two steps, a normalization and the scores layer, turn the last stream into scores. Applied to the stream after each earlier block, they show what the model would say if it stopped there.

## By hand

A board as 64 numbers. Squares run from a1 to h8. An empty square is 0, and a piece is 1 to 6 for white's pawn, knight, bishop, rook, queen and king, and 7 to 12 for black's.

| After | Square e2 | Square e4 | Square f3 | Square g1 |
|---|---|---|---|---|
| no moves | 1 | 0 | 0 | 2 |
| `e4` | 0 | 1 | 0 | 2 |
| `e4 e5 Nf3` | 0 | 1 | 2 | 0 |

The size of a probe. It maps 256 numbers to 64 × 13 = 832 scores: 212,992 weights and 832 biases.

The nudge. Say the probe's score for "empty" on a square trails its score for "queen" by 4. We want empty to lead by 3. Adding the line that separates the two to the stream raises that lead by the line's squared length. If the squared length is 2, we add (3 + 4) / 2 = 3.5 times the line.

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

This is an ordinary training loop on a model of one layer, with 256 positions drawn by chance for each step. `view(-1, PIECES)` lays the scores out as one row of 13 for each square of each position, so that lesson 3's loss can mark each square.

```python
# gpt/probe.py
def erase(probe, stream, square, piece, margin=3.0):
    """The smallest change to a stream, along the probe's own line, after which the probe scores
    "empty" on `square` above `piece`, by `margin`."""
    rows = probe.weight.view(64, PIECES, -1)[square]
    biases = probe.bias.view(64, PIECES)[square]
    line = rows[0] - rows[piece]  # toward "empty", away from the piece
    lead = (stream @ line + biases[0] - biases[piece]).item()  # how far "empty" leads now
    return (margin - lead) / (line @ line).item() * line.detach()
```

PyTorch stores the layer's grid with a row for each score (lesson 4), so `view(64, PIECES, -1)` sorts the rows by square and piece. `rows[0]` holds the probe's weights for "empty" on this square and `rows[piece]` those for the piece. Their difference is the one line in the stream along which the probe tells the two apart. `.item()` turns a tensor of one number into a plain number, and `.detach()` cuts the result loose from the slopes.

## The proof

```bash
uv run pytest tests/test_probe.py
```

- The hand table: the start, and the board after `e4 e5 Nf3`.
- The stream after the last block, put through the final layer, gives the model's own scores.
- A change made after one block changes the stream at the last token after that block and each later one, and at no earlier token.
- Each move's last token is found in a line of text, castling included.
- On made-up streams with a board written into them along straight lines, a probe reads over 0.9 of squares. Fitted to shuffled boards, it reads under 0.4.
- After `erase`, the probe scores empty above the piece on that square, by the margin asked for.

```bash
uv run python -m gpt.probe
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

Everything so far ran on one laptop: two models of about 5 million parameters, three hours each. We have not seen what the same code does on hardware built for it.

## Check yourself

Predict before you run: a probe fitted to shuffled boards has never been told what is on any board in front of it. How many squares will it still name right?

<details><summary>Answer</summary>

@@PREDICT@@
</details>

<details><summary>Why use one linear layer for the probe, and not a small network?</summary>

A network could work the board out by itself from whatever the stream holds, and we would learn about the network. One linear layer can do little more than read what is laid out for it.
</details>

<details><summary>The probe reads the board well. Why is that not enough to say the model uses a board?</summary>

The numbers could sit in the stream and play no part in the move. Changing them, and watching the move change the way the new board predicts, is the evidence that they are used.
</details>

<details><summary>From lesson 9: why is the stream after a block the right place to look?</summary>

Each block reads the stream and adds to it, and the final layer reads it to choose the move. Anything one block hands to a later one has to pass through it.
</details>

Go deeper:

- Li and others, [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382) (2022). The same experiment on the game of Othello, where it was first done.
- Karvonen, [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (2024). Linear probes and edits on a chess model.
- Alain and Bengio, [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644) (2016). Where probes come from.
