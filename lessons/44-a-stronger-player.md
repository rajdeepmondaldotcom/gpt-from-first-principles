# 44. A stronger player

By the end of this page the same model code has trained for 12 hours of steps on a rented card, on 19 million games between strong players, and you know each choice in that run, and its bill.

Lesson 41's chess model is rated +775 on a scale of our own, where a random mover is 0. We do not know what that is worth on a scale others use, or how strong this code can get.

## The idea

Lesson 41's model read four months of games between players of any rating. We change the games, the tokens, the words that open each game, and the model's size. The model's file stays as it is.

1. Both players of a game must be rated 2,200 or more on Lichess. We leave out bullet games, which Lichess expects to take each side under three minutes, and games it does not mark as ended normally, such as those a clock ended. We read fifteen months: 2,916 files, 999 GB.
2. A common move is one token. Lesson 5's cut never joins a letter to a digit, so lesson 38's moves took 2.1 tokens each. Here a chunk is anything between two spaces, and lesson 5's trainer stops when each chunk of its sample, 50,000 games, is one token.
3. A game's line starts with three words: white's rating in hundreds, black's, and the result, as in `W23 B22 1-0`. In training they are facts. In lesson 45 we write them, to ask for a strong player's moves. Chen and others steer a model this way, by the result they ask it for.
4. The model is 640 wide and 16 blocks deep, 76,286,080 parameters. Lesson 19's law was 0.347 nats off in lesson 41, so we did not size by it. Lesson 41's widest story model, 768 wide and 12 deep, failed lesson 13's check of the first loss: a wider model starts with wider scores (lesson 41). So this one is narrower, and deeper to stay near its size.

As in lesson 18, we drop repeats and near-repeats, month by month, and keep one game in a hundred unseen. The run is lesson 21's long run, in lesson 12's 16-bit numbers. Three short runs pick the rate, lesson 13's checks must pass, and each 500 steps the run scores unseen games and saves a checkpoint.

Two timings come before it. Lesson 6 showed `attend` equal to a routine of PyTorch's, which on an NVIDIA card may call faster code, such as lesson 17's tiles. The script times a step with each, and its output names PyTorch's routine the fused one: one call that does all of `attend`'s steps. Then it times steps of 32, 64 and 128 windows of 256 tokens, and trains at the size that puts the most tokens through in a second, for 12 hours of steps.

## By hand

Tokens, by lesson 38's cut and by this one:

| Move | Lesson 38 | Here |
|---|---|---|
| ` e4` | ` e`, `4` | ` e4` |
| ` Bxc6+` | ` Bxc`, `6`, `+` | ` Bxc6+` |
| ` O-O` | ` O`, `-`, `O` | ` O-O` |

A game of 100 moves was about 210 tokens. Here it is the end token, three words and 100 moves: 104.

The first loss. With a vocabulary of 6,556 tokens, a model that knows nothing scores ln 6,556 = 8.79 nats, and lesson 13's check wants the first loss within 0.15 of that.

The three words. White is rated 2,734, black 2,251, and black won. 2,734 // 100 = 27 and 2,251 // 100 = 22, so the line opens `W27 B22 0-1`.

## The code

```python
# gpt/strong.py
def told(white, black, result):
    """The three words that open a game's line: each player's rating in hundreds, and how the
    game ended. In training they are facts. At the board we choose them."""
    return f"W{white // 100} B{black // 100} {result}"


def strong(path, least, moves=20):
    """One line for each game in a file of Lichess games whose two players are both rated
    `least` or more: what `told` says of it, then lesson 38's line of its moves. Left out:
    bullet games, under three minutes a side, and games Lichess does not mark as ended normally."""
    wanted = (pc.field("WhiteElo") >= least) & (pc.field("BlackElo") >= least)
    wanted &= ~pc.match_substring(pc.field("Event"), "ullet") & (
        pc.field("Termination") == "Normal"
    )
    columns = ["WhiteElo", "BlackElo", "Result", "movetext"]
    table = pq.read_table(path, columns=columns, filters=wanted)
    for white, black, result, text in zip(*(table[name].to_pylist() for name in columns)):
        for line in games([text], moves):
            yield f"{told(white, black, result)} {line}"


# Lesson 5 cuts prose into runs of letters and runs of digits, so `Nf3` can never be one token.
# Here a chunk is a whole move with the space before it, and a common move becomes one token.
MOVES = re.compile(r" ?\S+|\s+")


def fused(q, k, v, mask=None):
    """Lesson 6's `attend`, done by PyTorch's own routine, which lesson 6's test found equal to
    it. On an NVIDIA card it may call faster code, such as lesson 17's tiles. It hands back no
    weights."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None
```

`pq` and `pc` are pyarrow, which reads Lichess's files: `read_table` takes from a file only the games that pass `wanted`. `"ullet"` is in the `Event` field of a bullet game, and `~` means not. `games` is lesson 38's, which keeps a game of 20 moves or more. `MOVES` matches a run of anything but spaces, with the space before it, and `|\s+` keeps any other run of spaces as a chunk of its own.

## The proof

```bash
uv run pytest tests/test_strong.py
```

- The filter leaves out a player rated 2,199, a rating not known, a bullet game and a game the clock ended.
- A common move is one token where lesson 5's cut makes it two, and a game comes back as it went in.
- With PyTorch's routine in `attend`'s place the model gives the same scores to within 0.00001, with lesson 23's cache too.

The runs need lesson 41's machine and its 200 GB disk: the games stage keeps one file at a time. We left out October 2024: its lines number a game's moves from 0, and lesson 38's `games` kept none.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/start.sh g6e.xlarge 1800
aws/run.sh <host> python -m gpt.strong games 2025-09 2025-08 2025-07 2025-06 2025-05 2025-04 2025-03 2025-02 2025-01 2024-12 2024-11 2024-09 2024-08 2024-07 2024-06
aws/run.sh <host> python -m gpt.strong ids
```

`ids` writes each month as token ids. We typed each stage's command on the machine itself, not through `aws/run.sh`, and split the months over three `games` commands. The two stages took 4 hours, beside lesson 42's second stage.

Predict before you run: a step of 128 windows holds four times the tokens of a step of 32. Will it put more tokens through in a second?

```bash
aws/run.sh <host> python -m gpt.strong train 12
```

```
games between players rated 2,200 or more: 19,449,873 read, 19,219,755 written for training, 1,548,506,981 tokens
tokens in the vocabulary: 6,556. Unseen games: 1.00 tokens a word
a training step of 76,286,080 parameters: 257 ms with lesson 6's attention, 97 ms with the fused one
32 windows a step: 97 ms, 84,095 tokens a second
64 windows a step: 219 ms, 74,668 tokens a second
128 windows a step: 518 ms, 63,206 tokens a second
step 500: train 5.101, unseen 5.003 nats
step 55500: train 1.739, unseen 1.733 nats
step 110500: train 1.697, unseen 1.673 nats
step 165500: train 1.629, unseen 1.626 nats
step 220500: train 1.579, unseen 1.584 nats
step 275500: train 1.510, unseen 1.537 nats
step 330500: train 1.485, unseen 1.493 nats
step 385500: train 1.433, unseen 1.452 nats
step 440500: train 1.434, unseen 1.431 nats
443,470 steps of 32 windows, 2.3 times through the games: 1.447 nats a token on unseen games, 12 hours of one card, $22.33
```

A window costs 3.0 ms in a step of 32 and 4.0 in a step of 128, so the run uses 32. As in lesson 41, a window cost more in a bigger step, and we did not find why.

The short runs are in `results/44_notes.json`: a peak rate of 0.0005 ended at 2.153 nats, 0.001 at 2.198 and 0.002 at 3.310. The lowest did best, and we tried none below it.

The run read the games 2.3 times and ends at 1.447 nats a token on all the unseen games: an even guess among e^1.447 = 4.3 tokens. Unseen games here take 1.00 tokens a word, where lesson 38's took 2.1 a move, so this loss does not compare with lesson 41's chess loss. The curve scores their first 100,000 tokens. Its last printed point, 1.431, is level with the training loss beside it, 1.434: no sign yet of games learned by heart.

The last line prices 12 hours of steps. On the clock the stage took 16, with the short runs first and a score and a checkpoint each 500 steps: about $30.

The script did not run lesson 18's cleaning on the unseen games, so 144 of their 193,852 repeat or nearly repeat another game, by the same file's count.

## What it still cannot do

We changed the games, the tokens, the three words and the size at once, where lesson 14 changes one thing at a time. Each such run is 16 hours of a card, and we do not know what each change bought. The bar of 2,200, the 12 hours and the model's shape were choices: we tested no other.

The model has a loss, and lesson 38 found that a loss says little about chess. It has not played a game, and we still have no opponent whose rating others know.

## Check yourself

<details><summary>Prediction</summary>

No. A step of 128 windows puts 63,206 tokens through in a second, and a step of 32 puts 84,095.
</details>

<details><summary>From lesson 13: why does the script check the first loss before the short runs, and not after the long one?</summary>

A first loss far from ln 6,556 means the starting weights are wrong, and lesson 41's widest story model failed this check. It needs no training step, and the run it guards takes 16 hours.
</details>

<details><summary>The line `W27 B22 0-1 e4 c5` follows an end token. How many tokens does the model read?</summary>

Six: the end token, three words and two moves.
</details>

Go deeper:

- Lichess, [its games on Hugging Face](https://huggingface.co/datasets/Lichess/standard-chess-games), and [how it names a game's speed](https://lichess.org/faq). The files the script reads, and the rule that makes a game bullet.
- Chen and others, [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) (2021). A model told the result to reach, and trained on recorded play.
- PyTorch, [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). The routine, and the faster ways of working it may choose on a card.
