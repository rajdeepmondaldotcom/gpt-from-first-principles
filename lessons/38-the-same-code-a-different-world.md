# 38. The same code, a different world

By the end of this page the model of lesson 21, with no line of it changed for chess, has learned the moves of chess from records of games, and you have counted how many of its moves the rules allow.

Each model in this course has read English: Shakespeare, then stories. We have not asked whether anything we built depends on that.

## The idea

Nothing in the model knows what a word is. It reads token ids and gives a chance to each id that could come next. We can put anything written as text in front of it, and chess players write their games down.

A game is a list of moves. `e4` moves a pawn to the square e4, `Nf3` a knight to f3, and `Bxc6` is a bishop taking on c6. `O-O` is castling, `+` is check and `#` is checkmate. White and black take turns, so a game reads `e4 e5 Nf3 Nc6`. On these pages a move is one side's turn, so that line has four. The model sees this line and never a board.

The games come from Lichess, a free site that gives the public a record of each rated game played on it. We take March 2014. `moves` strips a record down to its moves, and we keep each game of 20 moves or more.

From there the steps are lesson 18's: drop repeats and near-repeats, split by fingerprint, train lesson 5's tokenizer on a sample, write the ids to disk. Lesson 18's `clean` threw out a document under 80% letters and spaces, and a game is 74%. So we switch `clean` off for games: `moves` has done the cleaning.

The model is lesson 21's, with 636 tokens in place of 4,097: 4,488,448 parameters. The training is lesson 21's as well: that script's long run, called with other data.

A loss in nats says little about chess, and the rules give us an exact check of lesson 24's kind. Take a game the model has not seen and cut it at a random move. Hand the model the moves so far, let it write the next one, and ask a chess program whether that move is legal. What is legal depends on where each piece now stands, and the model is shown only the moves.

Many games open alike, so the moves so far may be the start of a game the model trained on. We count those apart.

## By hand

After `e4 e5` it is white's turn.

| Move | Legal | Why |
|---|---|---|
| `Nf3` | yes | the knight on g1 can jump to f3 |
| `Ke3` | no | a king moves one square |
| `e5` | no | black's pawn stands there |
| `Qh5` | yes | the pawn has left e2, so the queen's diagonal is open |

Before `e4`, `Qh5` was not legal: the pawn on e2 stood in the way.

Tokens. Lesson 5's tokenizer never joins a letter to a digit, so a move is two tokens or more.

| Move | Tokens |
|---|---|
| ` e4` | ` e`, `4` |
| ` Nf3` | ` Nf`, `3` |
| ` Bxc6+` | ` Bxc`, `6`, `+` |
| ` O-O` | ` O`, `-`, `O` |

The games come to 2.1 tokens a move, and half of them have 65 moves or fewer, about 140 tokens.

## The code

```python
# gpt/games.py
def moves(line):
    """One game's line from a Lichess file, as its moves alone: `e4 e5 Nf3 Nc6`."""
    return " ".join(NOT_A_MOVE.sub(" ", line).split())
```

`NOT_A_MOVE` is a pattern that matches what a record holds besides moves: notes in braces, move numbers, marks such as `?!`, and the result.

```python
# gpt/games.py
def legal(played, move, exact=False):
    """Whether `move` is a legal next move after the moves `played`, by the rules of chess.

    The library reads a move's marks loosely: `Nc3` passes where the rules write `Nxc3`, and so
    does `e4+` with no check. With `exact`, the move must also be written as the rules write it.
    """
    import chess  # the rules: a board that knows which moves are allowed

    board = chess.Board()
    for past in played:
        board.push_san(past)
    try:  # the library also reads `--` as a pass, which the rules do not allow
        found = board.parse_san(move)
    except ValueError:  # not a move, not possible here, or not clear which piece is meant
        return False
    return found in board.legal_moves and (not exact or board.san(found) == move)


def next_move(model, words, played, temperature=0, generator=None):
    """The move the model writes after the moves `played`: its tokens up to the next space.

    At a temperature of 0 that is its top move. Above 0 the move is drawn by chance.
    """
    ids = [words.end, *words.encode("".join(f" {move}" for move in played))]
    how = {"temperature": temperature, "generator": generator}
    written = generate(model, ids, 6, stop=words.end, **how)[len(ids) :]  # O-O-O+ is 6 tokens
    text = words.decode([i for i in written if i != words.end]).split()
    return text[0] if text else ""
```

`words` is the tokenizer and `generate` lesson 22's writing loop, told to write 6 tokens at most. The ids start with the end token because each game in the training text came after one. `chess` is python-chess, a library that knows the rules: the referee, and no part of the model. `push_san` plays a move on its board. `parse_san` raises an error for text that names no move the position allows, and reads `--` as a pass, so its answer is checked against the legal moves. `san` writes a move as the rules do.

## The proof

The rules are an extra: `uv sync --extra chess` installs them.

```bash
uv run pytest tests/test_games.py
```

- A record's line becomes moves alone, with notes, numbers, marks and results gone, and a record of 4 moves is left out. Each move kept is legal after the ones before it.
- The hand examples, and a few more: nothing is legal after checkmate, and `hello` and `--` are no moves. `Nf3+` with no check is a legal move, and not an exact one.
- A small model that knows one game by heart writes its next move at three places, castling among them.
- A cut falls inside a game's first 60 moves. A position is found among the training games it opens, and half a move opens none.

The script downloads 129 MB and trains on the graphics chip for 10,420 steps.

```bash
uv run python -m gpt.games
```

```
games read: 749,462, repeats: 18
near-repeats: 11, games kept: 749,433
train: 741,983 games, 111,113,359 tokens
val: 7,450 games, 1,116,979 tokens
tokens in the vocabulary: 636
unseen games: 2.1 tokens a move, half have 65 moves or fewer, 74% of a game is letters and spaces
parameters: 4,488,448, steps: 10,420, rate 0.001
before training: 0% of 200 moves are legal
step 500: unseen 1.990 nats, 58% of 200 moves are legal
step 1500: unseen 1.686 nats, 81% of 200 moves are legal
step 2500: unseen 1.578 nats, 84% of 200 moves are legal
step 3500: unseen 1.506 nats, 89% of 200 moves are legal
step 4500: unseen 1.459 nats, 88% of 200 moves are legal
step 5500: unseen 1.422 nats, 92% of 200 moves are legal
step 6500: unseen 1.390 nats, 89% of 200 moves are legal
step 7500: unseen 1.362 nats, 92% of 200 moves are legal
step 8500: unseen 1.340 nats, 91% of 200 moves are legal
step 9500: unseen 1.326 nats, 90% of 200 moves are legal
step 10420: unseen 1.317 nats, 92% of 200 moves are legal
after training: 929 of 1,000 moves are legal, 95% interval 0.911 to 0.944
of those, 884 are written with each mark as the rules write it
positions that open some training game: 192. Legal moves there: 192
positions no training game has: 808. Legal moves there: 737, 95% interval 0.890 to 0.931
of the 200 positions counted during training, 36 open some training game
a game it writes alone: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 h6 O-O Bc5 a3 a6 b4 Ba7 c3 d6 d4 exd4 cxd4 Bg4 Be3 O-O Qc2 Rfe8 h3 Bh5 Nd4 Nxd4 Bxd4 c5 dxc5 dxc5 Be3 b5 bxc5 Ng4 Bd4 Qh4 Bxh7+ Kxh7 Ng3 Qxh3 Nf5 Qf1+
its first 26 moves are legal, of 44
```

The month holds 749,462 games of 20 moves or more. 29 are repeats or near-repeats, and 7,450 of the rest are unseen.

Before training, none of 200 moves is legal. After 500 steps 58% are, and at the end 92%. On 1,000 fresh positions the model writes a legal move 929 times: 0.911 to 0.944. It writes 884 of them as the rules do. The other 45 name a legal move and write it another way.

Some of that could be memory. 192 of the 1,000 positions open some training game, and each move there is legal. The other 808 are lists of moves that no training game begins with, and 737 of their moves are legal: 0.890 to 0.931.

## What it still cannot do

```
e4 e5 Nf3 Nc6 Bc4 Nf6 d3 h6 O-O Bc5 a3 a6 b4 Ba7 c3 d6 d4 exd4 cxd4 Bg4 Be3 O-O Qc2 Rfe8 h3 Bh5 Nd4 Nxd4 Bxd4 c5 dxc5 dxc5 Be3 b5 bxc5 Ng4 Bd4 Qh4 Bxh7+ Kxh7 Ng3 Qxh3 Nf5 Qf1+
```

The model wrote that game alone, from the end token. Its first 26 moves are legal and the 27th is not.

A legal move can be a terrible one. We do not know yet whether the model can win a game.

## Check yourself

Predict before you run: the model has never seen a board. After 10,420 steps of reading games, what share of its moves will be legal?

<details><summary>Answer</summary>

About nine in ten. On lists of moves that no training game begins with, 737 of 808 moves are legal.
</details>

<details><summary>What changed in the model's file to make it play chess?</summary>

Nothing. The vocabulary of 636 is a number passed in. The data, the tokens and the check are new, and each lives outside the model.
</details>

<details><summary>From lesson 18: why split the games by fingerprint before training?</summary>

So that no game is on both sides. A game the model had trained on would be an easy test.
</details>

Go deeper:

- Karvonen, [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (2024). A larger model trained the same way, on 16 million games.
- Toshniwal and others, [Chess as a Testbed for Language Model State Tracking](https://arxiv.org/abs/2102.13249) (2021). Legal moves as a test of whether a model keeps track of a world.
- Lichess, [the open database](https://database.lichess.org/). Each rated game played on the site since 2013, in the public domain.
