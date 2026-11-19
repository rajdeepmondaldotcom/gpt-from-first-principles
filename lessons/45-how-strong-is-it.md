# 45. How strong is it?

By the end of this page lesson 44's model has played 1,300 games, and lesson 41's model 100, against a chess program held to known ratings, and has an Elo rating on that program's scale, with a range.

The model has a loss, and lesson 38 found that a loss says little about chess. It has not played a game, and we still have no opponent whose rating others know.

## The idea

Stockfish is a program that plays chess. One of its settings holds it to any rating we choose from 1,320 to 3,190, on a scale its authors fitted to a public list of rated programs.

The games follow lesson 39's rules, but stop after 240 moves, not 100: a common move is one token now, and the model reads 256. Stockfish gets 0.05 seconds a move, so 1,400 games fit in two hours. Its scale was fitted at a far slower clock, 120 seconds a game and a second more a move, so a last match gives it ten times as long, to see if that matters.

Lesson 39's player wrote a move, and the rules picked one when five tries failed. The new player plays the legal move the model scores highest, among those that are one token: one pass scores them all.

Lesson 44's three words are now ours to write, and we call what we write a telling: one rating for both players, and a win for the model's side or a draw. White in the unseen games is rated 2,200 or more, and the highest band of a hundred points that holds one game in a hundred is the 2,700s. We try 2,200, 2,400, 2,500 and 2,700 with a win, and 2,700 with a draw, each for 100 games against Stockfish held to 1,900. The best then plays new games.

Lesson 39 fitted every player's rating at once. Here we know Stockfish's seven ratings, and ours is the one unknown. The script keeps the whole-number rating below 4,000 that makes the 700 results most likely by the Elo rule, and lesson 24's bootstrap, 200 draws over pairs of games, gives it a range.

First, lesson 41's model plays the weakest setting, by lesson 39's rules.

## By hand

The Elo rule, turned round. A player who takes a share s of the points from an opponent rated S is rated

S + 400 × log10(s / (1 - s))

| s | s / (1 - s) | Rating |
|---|---|---|
| 0.50 | 1 | S |
| 0.76 | 3.2 | S + 200 |
| 0.20 | 0.25 | S - 241 |

One match gives one such rating. The fit gives one for all seven.

A telling: black is to move, and we tell the model that both players are rated 2,700 and its side wins. A black win is written 0-1, so the line opens `W27 B27 0-1`.

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

`words.vocab` maps a token's bytes to its id. UCI is the text protocol chess programs speak: `popen_uci` starts Stockfish, `UCI_LimitStrength` and `UCI_Elo` hold it to a rating, and `play` returns its move in `.move`. In `rating`, `mine` is a column of each rating from 0 to 3,999, and `likely` is the log of the chance each one gives the results.

## The proof

```bash
uv run pytest tests/test_strong.py
```

- Ten wins for each loss against a player rated 1,500 give a rating of 1,900, and a half share gives the opponent's own.
- Two opponents, each scored as a player rated 1,700 would score, give 1,700.
- The line the player hands the model opens with the end token and the three words, with `1-0` when it plays white and `0-1` when it plays black. It plays the one-token legal move scored highest.

Predict before you run: each game the model read was between players rated 2,200 or more. Will it take half the points from Stockfish held to 2,200?

The run is on lesson 44's machine, with Stockfish 17.1, its command typed there as before.

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

Lesson 41's model takes 0.23 of the points from Stockfish's weakest setting. The rule turned round makes that an Elo rating of about 1,110.

The five tellings score from 0.30 to 0.45, on openings of their own. The best owes part of that to the pick, so 100 games do not rank them.

The fit gives the new player an Elo rating of 1,785, with a range of 1,737 to 1,836, on Stockfish's scale, which is not Lichess's or a chess federation's. Our chess players:

| Player | Elo rating | Scale |
|---|---|---|
| A player that moves at random | 0 | ours: it is held at 0 |
| Lesson 38's model | +421 | ours, lesson 39's fit |
| The grabber, which looks one move ahead | +738 | ours, lesson 39's fit |
| Lesson 41's model | +775 | ours, lesson 41's fit, in which lesson 38's model is +355 |
| Lesson 41's model | about 1,110 | Stockfish's, one match, below its scale |
| Lesson 44's model | 1,785 | Stockfish's, seven matches, range 1,737 to 1,836 |

Only lesson 41's model is on both scales, and one match is too little to turn one into the other. On Stockfish's scale lesson 44's model stands about 1,785 - 1,110 = 675 points above lesson 41's, and lesson 44 cannot say which of its changes bought them.

One rating does not suit the seven matches. The same rule, on each match alone:

| Stockfish held to | Our share | The rule gives |
|---|---|---|
| 1,400 | 0.80 | 1,641 |
| 1,600 | 0.56 | 1,642 |
| 1,800 | 0.36 | 1,700 |
| 2,000 | 0.41 | 1,937 |
| 2,200 | 0.19 | 1,948 |
| 2,400 | 0.09 | 1,998 |
| 2,600 | 0.06 | 2,122 |

The stronger the setting, the higher the rating it gives us. Our shares put us 241 above the weakest setting and 478 below the strongest: 719 between the two, where their labels are 1,200 apart. The fit's range counts only the luck of the games, and we do not know why the matches disagree.

We tested one cause, the short clock. With 0.5 seconds a move, Stockfish held to 1,800 gave up 0.41 of the points where it had given 0.36. The intervals overlap: no sign that the short clock helped us there. That is still less time than the scale's own clock, and we did not try it at the stronger settings.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/stop.sh
```

Lessons 44 and 45 cost us $36.96 by the price list: `results/45_bill.json` has the hours. Three hours of the games stage ran beside lesson 42's second stage and are in its bill. With them, the runs took 22 hours, about $41.

## What it still cannot do

Lesson 44 bought 1,785 with 12 hours of steps and tried no rate below its best. We also changed how a move is picked, and never let the new model pick moves lesson 39's way. We did not tally how many games reached 240 moves and went to the piece count.

We have not taught lesson 42's bigger story model to take an order, and lesson 36's own server has not run on a card.

The player picks each move in one pass and never tries a reply on a board. [MAP.md](../MAP.md) has two ways on from here: a search through the moves ahead, and learning an engine's verdict on each move.

The README's tables hold each result, and [PRACTICE.md](../PRACTICE.md) has nine routines for practising what the lessons taught.

## Check yourself

<details><summary>Prediction</summary>

No: 0.19 of the points, with an interval of 0.12 to 0.27. Stockfish's 2,200 need not be Lichess's.
</details>

<details><summary>The best telling scored 0.45 against Stockfish held to 1,900. Why play it again in new games before fitting a rating?</summary>

We picked it for scoring highest of five, so part of that 0.45 is the luck that made it the pick. The new games had no part in the choice.
</details>

<details><summary>From lesson 39: that fit added a made-up drawn game between each two players. Why does this fit need none?</summary>

A fitted rating is finite unless one side took each point. Our player both won and lost against the seven settings, so one rating makes the results most likely.
</details>

Go deeper:

- Stockfish, [UCI and commands](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html). The setting that holds it to a rating, and the clock its scale was fitted at.
- Zhang and others, [Transcendence: Generative Models Can Outperform The Experts That Train Them](https://arxiv.org/abs/2406.11741) (2024). A chess model can sometimes play better than each player it learned from, and picking its likelier moves can enable that.
- Ruoss and others, [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494) (2024). A transformer taught an engine's verdict on each move, playing with no search.
