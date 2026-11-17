# 39. Can it play?

By the end of this page the chess model has played whole games against three opponents, and has a rating with a range around it.

A legal move can be a terrible one. We do not know yet whether the model can win a game.

## The idea

To find out whether something plays well, make it play. A game needs two players and a referee. The referee is the chess library of lesson 38: it holds the board, lists the legal moves and says when the game is over.

The model plays like this. It is handed the moves so far and writes its top move. If that move is not legal, it gets four more tries, each drawn by chance at a temperature of 1. If none of the five is legal, the referee picks a legal move for it at random, and we count a slip. Slips keep a game going, and the count of them says how much of the playing was the referee's.

A game stops after 100 moves, counting both sides. The model reads 256 tokens, and 100 moves are about 210. A game stopped there goes to the side that is ahead by 3 pawns or more in pieces, on the usual count: a pawn is 1, a knight or a bishop 3, a rook 5, a queen 9. Otherwise it is a draw.

It meets three opponents.

1. The random mover picks any legal move.
2. The grabber looks one move ahead. It gives checkmate if it can. If not, it takes the most valuable piece on offer. If there is nothing to take, it moves at random.
3. The model at step 2,500 is the same run a quarter of the way through its 10,420 steps, saved by lesson 38's script.

A win scores 1, a draw 0.5 and a loss 0. Each two of the four players meet for 60 games and swap colours each game, since white moves first. The share of the points a player took is a mean over games, so lesson 24's bootstrap gives it an interval, and lesson 24's bound stands in when one side took each point.

One number for strength is a rating. The Elo rule says what score to expect from two ratings: a player 400 points above another is expected to take 10 points for each 1 the other takes. We turn the rule round. Given the results, we look for the ratings that make those results most likely, which is lesson 1's idea of fitting, done with lesson 11's AdamW. The random mover is held at 0, so the scale is ours and is not the scale of a chess site. One made-up draw is added between each two players, so that a player who took each point still gets a rating that is not endless. Drawing the games again, with repeats, and fitting again gives each rating a range.

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

Being ahead. White has given up a knight and taken a rook and a pawn: 5 + 1 - 3 = 3. White is 3 ahead, enough to be given a game stopped at 100 moves.

## The code

```python
# gpt/play.py
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
```

A player is a function: it is given the board and the moves so far, and returns a move as text. `lead` is white's count of pieces minus black's, on the scale above. `board.turn` says whose move it is. `is_game_over` is true at checkmate and at each kind of draw the rules know.

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

- The shortest game there is leaves white mated after four moves and scores 0. A game in which white mates on the seventh move scores 1.
- A game cut off with white a pawn up is a draw. Cut off with white a queen and a pawn up, it is white's.
- The grabber mates when it can and takes a queen when one is offered.
- The random mover plays each of white's 20 first moves within 300 tries.
- Over 30 games with colours swapped, the grabber takes more than 0.6 of the points from the random mover.
- An untrained model still finishes a legal game, with nearly each of its moves picked by the referee.
- The Elo rule gives the three hand values. Ratings of 0, 200 and 500, used to make 6,000 results, are found again: the 200 to within 25 points and the 500 to within 35.

```bash
uv run python -m gpt.play
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

The model plays chess from a list of moves. It was never shown a board. A legal move depends on where each piece stands, and we have not looked for the pieces inside the model.

## Check yourself

Predict before you run: the grabber only ever looks one move ahead. Will three hours of reading games be enough for the model to beat it?

<details><summary>Answer</summary>

@@PREDICT@@
</details>

<details><summary>Player A is rated 200 above player B. In 100 games, about how many points should A expect?</summary>

76. The rule gives 1 / (1 + 10^-0.5) = 0.76 for each game.
</details>

<details><summary>Why does each pair swap colours from game to game?</summary>

White moves first, which is an edge. With colours swapped, each player has that edge in half of the games, and it cancels in the score.
</details>

<details><summary>From lesson 24: one player wins each of 60 games. Why can the bootstrap not give that score a range?</summary>

Each draw from 60 wins is 60 wins, so the interval has no width. Lesson 24 used another bound for that case, and this page uses it again. The made-up draw in the rating fit is there for the same reason.
</details>

Go deeper:

- Elo, [The Rating of Chessplayers, Past and Present](https://archive.org/details/ratingofchesspla00unse) (1978). The rule and the reasoning behind it, by the man it is named after.
- Karvonen, [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (2024). A larger model of the same kind, rated against a chess engine.
- Silver and others, [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (2017). What it takes to play chess well: search, and learning from the model's own games.
