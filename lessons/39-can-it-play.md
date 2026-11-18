# 39. Can it play?

By the end of this page the chess model has played whole games against three opponents, and has a rating with a range around it.

A legal move can be a terrible one. We do not yet know whether it can win a game.

## The idea

We make the model play whole games. The referee is lesson 38's chess library: it holds the board, lists the legal moves and says when the game is over.

The model is handed the moves so far and writes its top move. If that move is not legal, it gets four more tries, each drawn by chance at a temperature of 1. If none of the five is legal, the referee picks a legal move at random, and we count a slip.

A game stops after 100 moves, counting both sides: the model reads 256 tokens, and 100 moves are about 210. The game then goes to the side that is 3 pawns or more ahead in pieces, on the usual count: a pawn is 1, a knight or a bishop 3, a rook 5, a queen 9. Otherwise it is a draw.

It meets three opponents.

1. The random mover picks any legal move.
2. The grabber looks one move ahead: it gives checkmate if it can, or else takes the most valuable piece on offer, or else moves at random.
3. The model at step 2,500 is the same run a quarter of the way through its 10,420 steps.

A win scores 1, a draw 0.5 and a loss 0. Each two of the four players meet for 200 games, in 100 pairs. A pair starts from two opening moves of its own, drawn from the 400 there are, and is played twice with colours swapped, since white moves first. The share of the points is a mean over games, so lesson 24's bootstrap gives it an interval, drawn pair by pair.

One number for strength is a rating. The Elo rule says what score to expect from two ratings: a player 400 points above another is expected to take 10 points for each 1 the other takes. We turn the rule round: given the results, we look for the ratings that make them most likely. For a game with score s where the rule expected p, the loss is lesson 1's surprise, -(s ln p + (1 - s) ln(1 - p)), and lesson 11's AdamW lowers it.

We hold the random mover at 0, so these ratings do not compare with a chess site's. One made-up drawn game between each two players keeps a rating finite when one side won every game. Drawing the 600 pairs of games again and fitting again gives each rating a range.

## By hand

The Elo rule. The expected score of a player rated R against one rated S is

1 / (1 + 10^((S - R) / 400))

| R - S | Sum | Expected score |
|---|---|---|
| 0 | 1 / (1 + 1) | 0.50 |
| 200 | 1 / (1 + 10^-0.5) = 1 / 1.316 | 0.76 |
| 400 | 1 / (1 + 10^-1) = 1 / 1.1 | 0.91 |

A share of the points. 160 wins, 20 draws and 20 losses in 200 games:

(160 + 20 × 0.5) / 200 = 0.85

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

A player is a function: it is given the board and the moves so far, and returns a move as text. `lead` is white's count of pieces minus black's, on the scale above. `is_game_over` is true at checkmate, at stalemate, when neither side has the pieces to mate, and at the 75-move and fivefold-repetition draws.

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
- A match plays each opening twice with the colours swapped, and no opening twice over. Over 30 games the grabber takes more than 0.6 of the points from the random mover.
- A slip is counted when no try is legal, a pass is no move, and a failed first try is followed by tries drawn by chance.
- An untrained model still finishes a legal game: the referee picks nearly each of its moves.
- The Elo rule gives a half for equal ratings, and ten to one either way at 400 points. Ratings of 0, 200 and 500, used to make 6,000 results, are found again: the 200 to within 25 points and the 500 to within 35.

Predict before you run: the grabber looks one move ahead. Will 10,420 steps of reading games be enough for the model to beat it?

```bash
uv run python -m gpt.play
```

```
the grabber against the random mover: 0.98 of the points in 200 games, 95% interval 0.97 to 1.00
the model at step 2,500 against the random mover: 0.87 of the points in 200 games, 95% interval 0.82 to 0.91
the model against the random mover: 0.92 of the points in 200 games, 95% interval 0.88 to 0.95
the model at step 2,500 against the grabber: 0.05 of the points in 200 games, 95% interval 0.03 to 0.08
the model against the grabber: 0.16 of the points in 200 games, 95% interval 0.11 to 0.21
the model against the model at step 2,500: 0.64 of the points in 200 games, 95% interval 0.60 to 0.68
the model: the rules picked 961 of its 27,049 moves, 3.6%
the model at step 2,500: the rules picked 2,634 of its 27,236 moves, 9.7%
the grabber: rated +738 with the random mover at 0, 95% interval +690 to +798
the model at step 2,500: rated +310 with the random mover at 0, 95% interval +270 to +359
the model: rated +421 with the random mover at 0, 95% interval +378 to +465
```

The model takes 0.92 of the points from the random mover, and its younger self 0.87. The two intervals overlap, so this match does not separate them.

Against the grabber the model takes 0.16 of the points, where its younger self took 0.05. It takes 0.64 from that younger self, with an interval of 0.60 to 0.68, clear of a half.

The model slipped on 961 of its 27,049 turns, 3.6%, and its younger self on 9.7%.

With the random mover at 0, the grabber is rated +738, the model +421 and its younger self +310. The last two intervals, +378 to +465 and +270 to +359, do not meet.

## What it still cannot do

The model plays chess from a list of moves. We never showed it a board. A legal move depends on where each piece stands, and we have not looked for the pieces inside the model.

## Check yourself

<details><summary>Prediction</summary>

No. The model takes 0.16 of the points from the grabber. It has learned which moves are legal, and not yet which moves win.
</details>

<details><summary>Player A is rated 200 above player B. In 100 games, about how many points should A expect?</summary>

76. The rule gives 1 / (1 + 10^-0.5) = 0.76 for each game.
</details>

<details><summary>From lesson 24: one player wins each of 200 games. Why can the bootstrap not give that score a range?</summary>

Each draw from 200 wins is 200 wins, so the interval has no width. Lesson 24's exact range stands in, and the made-up draw in the rating fit is there for the same reason.
</details>

Go deeper:

- Elo, [The Rating of Chessplayers, Past and Present](https://archive.org/details/ratingofchesspla00unse) (1978). The rule and its reasoning, by its author.
- Silver and others, [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (2017). Chess played well: search, and learning from the model's own games.
