# 38. The same code, a different world

By the end of this page the model of lesson 21, with no line of it changed, has learned the moves of chess from records of games, and you have counted how many of its moves the rules allow.

Each model in this course has read English: Shakespeare, then stories. We have not asked whether anything we built depends on that.

## The idea

Nothing in the model knows what a word is. It reads the ids of tokens and gives a chance to each id that could come next. Anything that can be written as text can be put in front of it, and chess players write their games down as text.

A game is a list of moves. `e4` moves a pawn to the square e4: the letter is the column, a to h, and the digit is the row, 1 to 8. `Nf3` moves a knight to f3. `Bxc6` is a bishop taking the piece on c6. `O-O` is castling, `+` is check and `#` is checkmate. White and black take turns, so a game reads `e4 e5 Nf3 Nc6`. On these pages a move is one side's turn, so that line has four. The line is all the model will ever see: no board, no pieces and no rules.

The games come from Lichess, a free site that gives the record of each game played on it to the public. We take one month, March 2014. A record carries the players' names, the clocks and the result as well as the moves, and a function called `moves` keeps the moves alone. We keep each game of 20 moves or more.

From there the steps are lesson 18's, in the same order: drop repeats, drop near-repeats, split by fingerprint, train lesson 5's tokenizer on a sample, write the ids to disk. One thing had to give. Lesson 18 threw out a document as junk when under 80% of it was letters and spaces, and a game of chess is 74%. That one test is switched off for games.

The model is lesson 21's, with 636 tokens in place of 4,097: 4,488,448 parameters. The training is lesson 21's as well. That script's long run is now a function, and this lesson calls it with other data: three short runs to pick a rate, the two checks, then the same budget of arithmetic.

A loss in nats says little about chess, and the rules give us an exact check of lesson 24's kind. Take a game the model has not seen and cut it at a random move. Hand the model the moves so far, let it write the next one, and ask a chess program whether that move is legal on that board. A model that copied common moves would often be wrong, since what is legal depends on where each piece now stands. An untrained model writes 0 legal moves in 200.

## By hand

After `e4 e5` it is white's turn.

| Move | Legal | Why |
|---|---|---|
| `Nf3` | yes | the knight on g1 can jump to f3 |
| `Ke3` | no | a king moves one square, and e3 is two from e1 |
| `e5` | no | black's pawn stands there, and a pawn does not take straight ahead |
| `Qh5` | yes | the pawn has left e2, so the queen's diagonal is open |

Before `e4`, `Qh5` was not legal: the pawn on e2 stood in the queen's way. The same three characters are right or wrong by what came before them.

Tokens. Lesson 5's tokenizer never joins a letter to a digit, so a move is two tokens or more.

| Move | Tokens |
|---|---|
| ` e4` | ` e`, `4` |
| ` Nf3` | ` Nf`, `3` |
| ` Bxc6+` | ` Bxc`, `6`, `+` |
| ` O-O` | ` O`, `-`, `O` |

The games come to 2.1 tokens a move, and half of them have 66 moves or fewer, about 140 tokens.

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
def legal(played, move):
    """Whether `move` is a legal next move after the moves `played`, by the rules of chess."""
    import chess  # the rules: a board that knows which moves are allowed

    board = chess.Board()
    for past in played:
        board.push_san(past)
    try:
        board.parse_san(move)
    except ValueError:  # not a move, not possible here, or not clear which piece is meant
        return False
    return True


def next_move(model, words, played, temperature=0, generator=None):
    """The move the model writes after the moves `played`: its tokens up to the next space.

    At a temperature of 0 that is its top move. Above 0 the move is drawn by chance.
    """
    ids = [words.end, *words.encode("".join(f" {move}" for move in played))]
    how = {"temperature": temperature, "generator": generator}
    written = generate(model, ids, 6, stop=words.end, **how)[len(ids) :]  # O-O-O is 5 tokens
    text = words.decode([i for i in written if i != words.end]).split()
    return text[0] if text else ""
```

`words` is the tokenizer, as in each script since lesson 18, and `generate` is lesson 22's writing loop, told here to write 6 tokens at most. The ids start with the end token because each game in the training text came after one. `chess` is a library that knows the rules, called python-chess. It is the referee here and no part of the model. `push_san` plays a move on its board, and `parse_san` raises an error for a move the position does not allow.

## The proof

The rules are an extra: `uv sync --extra chess` installs them.

```bash
uv run pytest tests/test_games.py
```

- A record's line becomes moves alone, with notes, numbers, marks and results gone.
- Of three records, the one with 4 moves is left out. Each move of the other two is legal after the moves before it.
- The hand examples, and a few more: nothing is legal after checkmate, and `hello` is no move.
- An untrained model's answer is cut at the first space, and each answer is marked legal or not.

The script downloads 129 MB, and trains for about three hours on the graphics chip.

```bash
uv run python -m gpt.games
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

```
@@SAMPLE@@
```

@@SAMPLETEXT@@

A legal move can be a terrible one. We do not know yet whether the model can win a game.

## Check yourself

Predict before you run: the model has never seen a board. After three hours of reading games, what share of its moves will be legal?

<details><summary>Answer</summary>

@@PREDICT@@
</details>

<details><summary>What changed in the model's file to make it play chess?</summary>

Nothing. The model is built with a vocabulary of 636 in place of 4,097, which is a number passed in. The data, the tokens and the check are new, and each lives outside the model.
</details>

<details><summary>From lesson 18: why split the games by fingerprint before training?</summary>

So that no game is on both sides. A game the model had trained on would be an easy test, and the count of legal moves would look better than it is.
</details>

Go deeper:

- Karvonen, [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (2024). A larger model trained the same way, on 16 million games, and how well it plays.
- Toshniwal and others, [Chess as a Testbed for Language Model State Tracking](https://arxiv.org/abs/2102.13249) (2021). Legal moves as a test of whether a model keeps track of a world.
- Lichess, [the open database](https://database.lichess.org/). Each rated game played on the site since 2013, in the public domain.
