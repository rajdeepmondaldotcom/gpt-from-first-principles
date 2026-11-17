# 39. Can it play?

By the end of this page the chess model has played whole games against three opponents, and has a rating with a range around it.

A legal move can be a terrible one. We do not know yet whether the model can win a game.

## The idea

To find out whether something plays well, make it play. The referee is lesson 38's chess library: it holds the board, lists the legal moves and says when the game is over.

The model is handed the moves so far and writes its top move. If that move is not legal, it gets four more tries, each drawn by chance at a temperature of 1. If none of the five is legal, the referee picks a legal move at random, and we count a slip.

A game stops after 100 moves, counting both sides: the model reads 256 tokens, and 100 moves are about 210. It then goes to the side that is 3 pawns or more ahead in pieces, on the usual count: a pawn is 1, a knight or a bishop 3, a rook 5, a queen 9. Otherwise it is a draw.

It meets three opponents.

1. The random mover picks any legal move.
2. The grabber looks one move ahead: it gives checkmate if it can, or else takes the most valuable piece on offer, or else moves at random.
3. The model at step 2,500 is the same run a quarter of the way through its 10,420 steps.

A win scores 1, a draw 0.5 and a loss 0. Each two of the four players meet for 60 games, in 30 pairs. A pair starts from two moves picked at random, so no two pairs are the same game, and is played twice with colours swapped, since white moves first. The share of the points is a mean over games, so lesson 24's bootstrap gives it an interval, drawn pair by pair.

One number for strength is a rating. The Elo rule says what score to expect from two ratings: a player 400 points above another is expected to take 10 points for each 1 the other takes. We turn the rule round: given the results, we look for the ratings that make them most likely, which is lesson 1's fitting done with lesson 11's AdamW. The random mover is held at 0, so the scale is ours and not a chess site's. One made-up draw between each two players keeps a rating finite when one side took each point. Drawing the pairs of games again and fitting again gives each rating a range.

## By hand

The Elo rule. The expected score of a player rated R against one rated S is

1 / (1 + 10^((S - R) / 400))

| R - S | Sum | Expected score |
|---|---|---|
| 0 | 1 / (1 + 1) | 0.50 |
| 200 | 1 / (1 + 10^-0.5) = 1 / 1.316 | 0.76 |
| 400 | 1 / (1 + 10^-1) = 1 / 1.1 | 0.91 |

A share of the points. 45 wins, 10 draws and 5 losses in 60 games:

(45 + 10 × 0.5) / 60 = 0.83

Being ahead. White has given up a knight and taken a rook and a pawn: 5 + 1 - 3 = 3, enough to be given a stopped game.

## The code

```python
# gpt/play.py
def result(board):
    """White's score for a game that stopped at this board: 1, 0, or 0.5 for a draw.

    A game the rules ended is scored by the rules. One stopped at the move limit goes to a side
    that is 3 pawns or more ahead.
    """
    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0  # the side to move has been mated
    ahead = 0 if board.is_game_over() else lead(board)
    return 1.0 if ahead >= 3 else 0.0 if ahead <= -3 else 0.5


def play(white, black, opening=(), most=100):
    """One game from the moves `opening`. Returns white's score.

    It stops after `most` moves, counting both sides: the model reads 256 tokens, and 100 moves
    are about 210.
    """
    board, played = chess.Board(), list(opening)
    for move in opening:
        board.push_san(move)
    while not board.is_game_over() and len(played) < most:
        move = (white if board.turn == chess.WHITE else black)(board, played)
        board.push_san(move)
        played.append(move)
    return result(board)
```

A player is a function: it is given the board and the moves so far, and returns a move as text. `lead` is white's count of pieces minus black's, on the scale above. `is_game_over` is true at checkmate and at each kind of draw the rules know.

```python
# gpt/play.py
def expected(mine, theirs):
    """The score the Elo rule expects for a player rated `mine` against one rated `theirs`."""
    return 1 / (1 + 10 ** ((theirs - mine) / 400))
```

## The proof

```bash
uv run pytest tests/test_play.py
```

- The shortest game there is leaves white mated after four moves and scores 0. White mating on the seventh move scores 1.
- A stopped game with white two pawns ahead is a draw, and with three it is white's. A stalemate is a draw though white is a queen ahead.
- A game can start from moves already played.
- The grabber mates when it can and takes a queen when one is offered.
- A match plays each opening twice with the colours swapped. Over 30 games the grabber takes more than 0.6 of the points from the random mover.
- A slip is counted when no try is legal, and a pass is no move.
- An untrained model still finishes a legal game: the referee picks nearly each of its moves.
- The Elo rule gives the three hand values. Ratings of 0, 200 and 500, used to make 6,000 results, are found again: the 200 to within 25 points and the 500 to within 35.

```bash
uv run python -m gpt.play
```

```
the grabber against the random mover: 0.98 of the points in 60 games, 95% interval 0.96 to 1.00
the model at step 2,500 against the random mover: 0.87 of the points in 60 games, 95% interval 0.79 to 0.93
the model against the random mover: 0.82 of the points in 60 games, 95% interval 0.75 to 0.90
the model at step 2,500 against the grabber: 0.05 of the points in 60 games, 95% interval 0.01 to 0.11
the model against the grabber: 0.23 of the points in 60 games, 95% interval 0.13 to 0.33
the model against the model at step 2,500: 0.62 of the points in 60 games, 95% interval 0.53 to 0.71
the model: the rules picked 278 of its 8,017 moves, 3.5%
the model at step 2,500: the rules picked 794 of its 8,266 moves, 9.6%
the grabber: rated +608 with the random mover at 0, 95% interval +538 to +738
the model at step 2,500: rated +250 with the random mover at 0, 95% interval +202 to +335
the model: rated +340 with the random mover at 0, 95% interval +267 to +439
```

The model takes 0.82 of the points from the random mover. At step 2,500 it took 0.87, and the two intervals overlap: against this opponent, more training does not show.

Against the grabber it does. The model takes 0.23 of the points, where its younger self took 0.05. And it takes 0.62 from that younger self, with an interval of 0.53 to 0.71, clear of a half.

The referee moved for the model in 278 of its 8,017 turns, 3.5%, and for its younger self in 9.6%.

On one scale, with the random mover at 0, the grabber is at +608, the model at +340 with an interval of +267 to +439, and its younger self at +250. The model beats a player that moves at random. A player that looks one move ahead takes three points in four from it.

## What it still cannot do

The model plays chess from a list of moves. It was never shown a board. A legal move depends on where each piece stands, and we have not looked for the pieces inside the model.

## Check yourself

Predict before you run: the grabber only ever looks one move ahead. Will 10,420 steps of reading games be enough for the model to beat it?

<details><summary>Answer</summary>

No. The model takes 0.23 of the points from the grabber. It has learned which moves are legal. Which moves win is another thing to learn.
</details>

<details><summary>Player A is rated 200 above player B. In 100 games, about how many points should A expect?</summary>

76. The rule gives 1 / (1 + 10^-0.5) = 0.76 for each game.
</details>

<details><summary>From lesson 24: one player wins each of 60 games. Why can the bootstrap not give that score a range?</summary>

Each draw from 60 wins is 60 wins, so the interval has no width. Lesson 24's exact range stands in, and the made-up draw in the rating fit is there for the same reason.
</details>

Go deeper:

- Elo, [The Rating of Chessplayers, Past and Present](https://archive.org/details/ratingofchesspla00unse) (1978). The rule and its reasoning, by its author.
- Karvonen, [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (2024). A larger model of the same kind, rated against a chess engine.
- Silver and others, [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (2017). Chess played well: search, and learning from the model's own games.
