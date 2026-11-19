# 45. How strong is it?

By the end of this page lesson 44's model has played 1,400 games against a chess program held to known ratings, and has a rating on that program's scale, with a range around it.

The model has a loss, and lesson 38 found that a loss says little about chess. It has not played a game, and we still have no opponent whose rating others know.

## The idea

Stockfish is a program that plays chess. One of its settings holds it to any rating we choose from 1,320 to 3,190, on a scale its authors fitted to a public list of rated programs.

The games follow lesson 39's rules, but stop after 240 moves, not 100: a move is one token now, and the model reads 256. Stockfish gets 0.05 seconds a move, so that 1,400 games fit in two hours. Its scale was fitted at 120 seconds a game and a second more a move, so a last match gives it ten times as long.

Lesson 39's player wrote a move, and the rules picked one when five tries failed. The new player plays the legal move the model scores highest, among those that are one token.

Lesson 44's three words are now ours to write, and we call what we write a telling: one rating for both players, and a win for the model's side or a draw. White's ratings in the unseen games start at 2,200, and the 2,700s are the highest hundred that holds one game in a hundred. We try four ratings from 2,200 to 2,700 with a win, and 2,700 with a draw, each for 100 games against Stockfish held to 1,900. The best then plays new games: we picked it for its score, so part of that score is the luck of the pick.

For the rating, lesson 39 fitted each player's at once. Here Stockfish's seven ratings are given, and ours is the one unknown. The script keeps the whole rating below 4,000 that makes the 700 results most likely by the Elo rule, and lesson 24's bootstrap, over pairs of games, gives it a range.

First, lesson 41's model plays the weakest setting, with lesson 39's player and its 100 moves.

## By hand

The Elo rule, turned round. A player who takes a share s of the points from an opponent rated S is rated

S + 400 × log10(s / (1 - s))

| s | s / (1 - s) | Rating |
|---|---|---|
| 0.50 | 1 | S |
| 0.76 | 3.2 | S + 200 |
| 0.20 | 0.25 | S - 241 |

One match gives one such rating, and the fit one rating for all of them.

## The code

```python
# gpt/strong.py
def best_legal(model, words, rated, wins=True):
    """A player for lesson 39's games. Of the legal moves that are one token, it plays the one
    the model gives the highest chance, and if none is one token, the first the rules list. The
    model is told that both players are rated `rated`, and that its own side wins, or a draw."""
    import torch

    def move(board, played):
        result = "1/2-1/2" if not wins else "1-0" if board.turn else "0-1"  # white is to move
        opening = told(rated, rated, result).split()
        ids = [words.end, *words.encode("".join(f" {one}" for one in (*opening, *played)))]
        # shortcut: a line longer than the context loses its start, the three words with it. A
        # game of 240 one-token moves is 244 tokens. Keep the words apart if games get longer.
        ids = torch.tensor([ids[-model.context :]], device=model.embed.weight.device)
        with torch.no_grad():
            scores = model(ids)[0, -1]
        legal = [board.san(one) for one in board.legal_moves]
        known = [san for san in legal if f" {san}".encode() in words.vocab]
        return max(known or legal, key=lambda san: scores[words.vocab.get(f" {san}".encode(), 0)])

    return move


def stockfish(elo, seconds=0.05, path="stockfish"):
    """Stockfish, a chess engine, held to a rating and given `seconds` for a move. Returns a
    player for lesson 39's games, and the engine, to quit when the games are over."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def move(board, played):
        return board.san(engine.play(board, chess.engine.Limit(time=seconds)).move)

    return move, engine


def rating(results):
    """The rating that makes these results most likely by lesson 39's Elo rule, to the nearest
    point. `results` holds, for each game, the opponent's rating and our score."""
    import torch

    from gpt.play import expected

    theirs = torch.tensor([float(their) for their, _ in results])
    score = torch.tensor([float(ours) for _, ours in results])
    mine = torch.arange(0.0, 4000.0)[:, None]  # each rating we might have, tried in turn
    guess = expected(mine, theirs).clamp(1e-6, 1 - 1e-6)
    likely = (score * guess.log() + (1 - score) * (1 - guess).log()).sum(dim=1)
    return likely.argmax().item()
```

`words.vocab` maps a token's bytes to its id. `UCI_Elo` is the setting that holds Stockfish to a rating. In `rating`, `expected` is lesson 39's Elo rule, `mine` is a column of each rating from 0 to 3,999, and `likely` is the log of the chance each one gives the results.

## The proof

```bash
uv run pytest tests/test_strong.py
```

- Ten points for each one lost to a player rated 1,500 is a rating of 1,900, and a half share is the opponent's own rating.
- Two opponents, each scored as a player rated 1,700 would score, give 1,700.
- The line the player hands the model opens with the end token and the three words, with `1-0` when it plays white and `0-1` when it plays black. It plays the legal move scored highest.

Predict before you run: each game the model read was between players rated 2,200 or more. Will it take half the points from Stockfish held to 2,200?

The run is on lesson 44's machine, with Stockfish 17.1.

```bash
aws/run.sh <host> 'sudo apt-get install -y stockfish && python -m gpt.strong play'
```

```
lesson 41's chess model:
against Stockfish held to 1,320, 0.05 s a move: 0.23 of the points in 100 games, 95% interval 0.15 to 0.31
white's rating in the unseen games starts at 2,200. The highest hundred that one game in a hundred has is 2,700
the stronger player, told: both players rated 2,200, and its own side wins
against Stockfish held to 1,900, 0.05 s a move: 0.41 of the points in 100 games, 95% interval 0.31 to 0.50
the stronger player, told: both players rated 2,400, and its own side wins
against Stockfish held to 1,900, 0.05 s a move: 0.32 of the points in 100 games, 95% interval 0.23 to 0.40
the stronger player, told: both players rated 2,500, and its own side wins
against Stockfish held to 1,900, 0.05 s a move: 0.36 of the points in 100 games, 95% interval 0.28 to 0.44
the stronger player, told: both players rated 2,700, and its own side wins
against Stockfish held to 1,900, 0.05 s a move: 0.45 of the points in 100 games, 95% interval 0.35 to 0.56
the stronger player, told: both players rated 2,700, and a draw
against Stockfish held to 1,900, 0.05 s a move: 0.30 of the points in 100 games, 95% interval 0.22 to 0.39
the stronger player in new games, told: both players rated 2,700, and its own side wins
against Stockfish held to 1,400, 0.05 s a move: 0.80 of the points in 100 games, 95% interval 0.73 to 0.88
against Stockfish held to 1,600, 0.05 s a move: 0.56 of the points in 100 games, 95% interval 0.47 to 0.65
against Stockfish held to 1,800, 0.05 s a move: 0.36 of the points in 100 games, 95% interval 0.28 to 0.46
against Stockfish held to 2,000, 0.05 s a move: 0.41 of the points in 100 games, 95% interval 0.31 to 0.52
against Stockfish held to 2,200, 0.05 s a move: 0.19 of the points in 100 games, 95% interval 0.12 to 0.27
against Stockfish held to 2,400, 0.05 s a move: 0.09 of the points in 100 games, 95% interval 0.04 to 0.14
against Stockfish held to 2,600, 0.05 s a move: 0.06 of the points in 100 games, 95% interval 0.03 to 0.10
the stronger player is rated 1,785 on Stockfish's scale, 95% interval 1,737 to 1,836
against Stockfish held to 1,800, 0.5 s a move: 0.41 of the points in 100 games, 95% interval 0.32 to 0.50
```

Lesson 41's model takes 0.23 of the points from Stockfish's weakest setting. By hand's rule makes that about 1,110, below the lowest rating the scale has, so no setting can check it.

The five tellings score from 0.30 to 0.45, and each range overlaps the best one's: 100 games do not tell them apart.

The fit gives 1,785, with a range of 1,737 to 1,836. But one rating does not suit the seven matches. By hand's rule, on each match alone:

| Stockfish held to | Our share | The rule gives |
|---|---|---|
| 1,400 | 0.80 | 1,641 |
| 1,600 | 0.56 | 1,642 |
| 1,800 | 0.36 | 1,700 |
| 2,000 | 0.41 | 1,937 |
| 2,200 | 0.19 | 1,948 |
| 2,400 | 0.09 | 1,998 |
| 2,600 | 0.06 | 2,122 |

The stronger the setting, the higher the rating it gives us: the two end settings are 1,200 points apart, and by the rule our shares put them 719 apart. The fit's range counts only the luck of the games, and we do not know why the matches disagree.

We tested one possible cause. With 0.5 seconds a move, Stockfish held to 1,800 gave up 0.41 of the points where it had given 0.36. The ranges overlap: no sign that the short clock helped us there. We did not try it at the stronger settings.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/stop.sh
```

Lessons 44 and 45 cost $36.96 by the price list, and `results/45_bill.json` has the hours.

## What it still cannot do

1,785 is what 12 hours of steps bought, not this code's limit: lesson 44 tried no rate below its best. We also changed how a move is picked, and never played lesson 39's player with the new model. We did not count the games that the 240-move limit stopped.

The player picks each move in one pass and never tries a reply on a board. [MAP.md](../MAP.md) has two ways on from here: a search through the moves ahead, and learning an engine's verdict on each position.

This is the last lesson. [PRACTICE.md](../PRACTICE.md) has routines for the skills that reading cannot give you.

## Check yourself

<details><summary>Prediction</summary>

No: 0.19 of the points, with a range of 0.12 to 0.27. Stockfish's scale is anchored to a list of programs and not to Lichess, so its 2,200 need not be Lichess's.
</details>

<details><summary>From lesson 39: that fit added a made-up drawn game between each two players. Why does this fit need none?</summary>

A rating runs off only when one side takes each point. Our player both won and lost against the seven settings, so one rating makes the results most likely.
</details>

<details><summary>The best telling scored 0.45 against Stockfish held to 1,900. Why play it again in new games before fitting a rating?</summary>

It was picked for scoring highest of five, so part of that 0.45 is the luck that made it the pick. The new games had no part in the choice.
</details>

Go deeper:

- Stockfish, [UCI and commands](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html). The setting that holds it to a rating, and the clock its scale was fitted at.
- Zhang and others, [Transcendence: Generative Models Can Outperform The Experts That Train Them](https://arxiv.org/abs/2406.11741) (2024). A chess model can sometimes play better than each player it learned from, and picking its likelier moves can enable that.
