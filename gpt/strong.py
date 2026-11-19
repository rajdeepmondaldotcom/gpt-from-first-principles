"""Lesson 44. A stronger player: learn only from strong players' games, and measure the result
against an engine whose rating the world knows."""

import re
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch.nn.functional as F

from gpt.games import games

# Lichess's rated games as it publishes them on Hugging Face: a folder of files for each month.
LISTING = "https://huggingface.co/api/datasets/Lichess/standard-chess-games/tree/main/data/year={}/month={}"
FILE = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/"


def told(white, black, result):
    """The three words that open a game's line: each player's rating in hundreds, and how the
    game ended. In training they are facts. At the board we choose them."""
    return f"W{white // 100} B{black // 100} {result}"


def strong(path, least, moves=20):
    """One line for each game in a file of Lichess games whose two players are both rated
    `least` or more: what `told` says of it, then lesson 38's line of its moves. Left out:
    bullet games, a minute or two a side, and games that a clock or a lost connection ended."""
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
    """Lesson 6's `attend`, done by PyTorch's own routine. On a card it may work in lesson 17's
    tiles and never keep the grid of scores: less memory and less time for the same numbers. It
    hands back no weights."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None


def best_legal(model, words, rated, wins=True):
    """A player for lesson 39's games. Of the legal moves that are one token, it plays the one
    the model gives the highest chance, and if none is one token, the first the rules list. The
    model is told that both players are rated `rated`, and that its own side wins, or a draw."""
    import torch

    def move(board, played):
        result = "1/2-1/2" if not wins else "1-0" if board.turn else "0-1"  # white is to move
        opening = told(rated, rated, result).split()
        ids = [words.end, *words.encode("".join(f" {one}" for one in (*opening, *played)))]
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


if __name__ == "__main__" and sys.argv[1:2] == ["games"]:
    # python -m gpt.strong games 2025-09 2025-08: each month's strong games, one to a line.
    # A month is 66 files of 1 GB. Each is read as soon as it is down, and then removed.
    import json
    import urllib.request
    from pathlib import Path

    cache = Path("data/cache")
    for month in sys.argv[2:]:
        out, kept = cache / f"strong_{month}.txt", 0
        if out.exists():
            continue
        with urllib.request.urlopen(LISTING.format(*month.split("-"))) as listing:
            files = [entry["path"] for entry in json.load(listing)]
        with out.with_suffix(".part").open("w") as file:
            for path in files:
                urllib.request.urlretrieve(FILE + path, out.with_suffix(".parquet"))
                for game in strong(out.with_suffix(".parquet"), 2200):
                    file.write(game + "\n")
                    kept += 1
        out.with_suffix(".parquet").unlink()
        out.with_suffix(".part").rename(out)  # a month cut short leaves no file by this name
        print(f"{month}: {kept:,} games between players rated 2,200 or more", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["ids"]:
    # python -m gpt.strong ids: the strong games as token ids, a file for each month. With
    # months named after it, only those. The first run trains the vocabulary on its first month,
    # the newest if none is named. Once that file is there, several can run side by side.
    import json
    from pathlib import Path

    from gpt.bpe import train
    from gpt.data import held_out, tokenizer
    from gpt.gpu import more

    cache = Path("data/cache")
    months = [cache / f"strong_{month}.txt" for month in sys.argv[2:]]
    months = months or sorted(cache.glob("strong_20*.txt"), reverse=True)
    if not (cache / "strong_vocab.json").exists():  # lesson 5's trainer, on 50,000 games
        sample = months[0].read_text().split("\n")[:50_000]
        vocab = train(" " + " ".join(sample), 8192, MOVES)
        (cache / "strong_vocab.json").write_text(json.dumps([token.hex() for token in vocab]))
    words = tokenizer(cache / "strong_vocab.json")
    words.split = MOVES
    for month in months:  # lesson 18's steps: drop repeats, keep the unseen games out
        docs = [game for game in month.read_text().split("\n") if game]
        counts = more(month.with_suffix(".bin"), docs, words, tidy=lambda game: game, lead=" ")
        # shortcut: taken from the month as read, so the few repeats that `more` drops from
        # training stay in here. Take them from what `prepare` keeps to be exact.
        unseen = [game for game in docs if held_out(game)]
        month.with_suffix(".val").write_text("\n".join(unseen) + "\n")
        month.with_suffix(".json").write_text(json.dumps(counts))
        print(f"{month.stem}: {counts['written']:,} games written", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["train"]:
    # python -m gpt.strong train 12: train on the strong games for that many hours of this card.
    import json
    from pathlib import Path

    import torch

    import gpt.model
    from gpt.checks import starts_with_no_opinion
    from gpt.cost import forward_flops
    from gpt.data import read, tokenizer, write
    from gpt.gpt2 import evaluate
    from gpt.gpu import PRICE, step_seconds
    from gpt.model import GPT
    from gpt.results import save, say
    from gpt.scale import shape
    from gpt.story import long_run, save_llama

    cache, hours = Path("data/cache"), float(sys.argv[2])
    months = sorted(cache.glob("strong_20*.bin"))
    assert months, "run python -m gpt.strong games and then python -m gpt.strong ids first"
    words = tokenizer(cache / "strong_vocab.json")
    words.split = MOVES
    counts = [json.loads(month.with_suffix(".json").read_text()) for month in months]
    unseen = "".join(month.with_suffix(".val").read_text() for month in months).split("\n")
    unseen = [game for game in unseen if game]
    if not (cache / "strong_val.bin").exists():
        write(
            cache / "strong_val.bin",
            [i for game in unseen for i in (*words.encode(" " + game), words.end)],
        )
    ids = torch.cat([read(month) for month in months]).to("cuda")
    val_ids = read(cache / "strong_val.bin").to("cuda")
    spoken = sum(len(game.split()) for game in unseen)
    say(
        f"games between players rated 2,200 or more: {sum(c['read'] for c in counts):,} read, "
        f"{sum(c['written'] for c in counts):,} written for training, {len(ids):,} tokens"
    )
    say(
        f"tokens in the vocabulary: {words.end + 1:,}. Unseen games: "
        f"{(len(val_ids) - len(unseen)) / spoken:.2f} tokens a word"
    )

    # The same step with lesson 6's attention and with PyTorch's fused routine in its place,
    # then with more windows in a step. We train at the size that gets the most tokens through.
    # Lesson 41's widest shape, 768 wide, failed lesson 13's first check. This one is narrower
    # and deeper, and the check runs here, before an hour goes on timing and short runs.
    form = shape(640, 16) | {"vocab": words.end + 1}
    torch.manual_seed(0)
    model = GPT(**form).to("cuda")
    assert starts_with_no_opinion(model, val_ids[:20_000], form["vocab"])
    ours = step_seconds(model, ids, half=True)
    gpt.model.attend = fused
    took = {windows: step_seconds(model, ids, windows, half=True) for windows in (32, 64, 128)}
    say(
        f"a training step of {sum(p.numel() for p in model.parameters()):,} parameters: "
        f"{ours * 1e3:.0f} ms with lesson 6's attention, {took[32] * 1e3:.0f} ms with the fused one"
    )
    for windows, seconds in took.items():
        say(
            f"{windows} windows a step: {seconds * 1e3:.0f} ms, "
            f"{windows * model.context / seconds:,.0f} tokens a second"
        )
    windows = max(took, key=lambda windows: windows / took[windows])  # the most tokens a second
    work = 3 * forward_flops(model, windows * model.context, model.context)
    del model

    budget = hours * 3600 * work / took[windows]
    model, notes, steps, _ = long_run(
        cache,
        "strong",
        form,
        ids,
        val_ids,
        budget=budget,
        check=budget / 100,
        half=True,
        size=windows,
    )
    nats = evaluate(model, val_ids)
    for point in notes["curve"][:: max(1, len(notes["curve"]) // 8)]:
        say(f"step {point['step']}: train {point['train']:.3f}, unseen {point['val']:.3f} nats")
    card = torch.cuda.get_device_name(0)
    say(
        f"{steps:,} steps of {windows} windows, "
        f"{steps * windows * form['context'] / len(ids):.1f} times through the games: "
        f"{nats:.3f} nats a token on unseen games, {hours:.0f} hours of one card, "
        f"${hours * PRICE[card][1]:.2f}"
    )
    out = cache / "strong"
    out.mkdir(exist_ok=True)
    save_llama(model.to("cpu"), out / "model.safetensors")
    (out / "vocab.json").write_text((cache / "strong_vocab.json").read_text())
    (out / "shape.json").write_text(json.dumps(form))
    save(
        "44_strong",
        torch=torch.__version__,
        device=card,
        counts=counts,
        seconds_a_step={"ours": ours} | {str(windows): took[windows] for windows in took},
        windows=windows,
        steps=steps,
        val_nats=nats,
        curve=notes["curve"],
    )


if __name__ == "__main__" and sys.argv[1:2] == ["play"]:
    # python -m gpt.strong play: the stronger player against Stockfish held to seven ratings.
    import shutil
    from pathlib import Path

    import torch

    import gpt.model
    from gpt.evaluate import interval, rate_interval
    from gpt.play import match, model_mover
    from gpt.results import save, say
    from gpt.story import load

    engine_path = shutil.which("stockfish") or "/usr/games/stockfish"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, generator = Path("data/cache"), torch.Generator().manual_seed(0)
    gpt.model.attend = fused

    def against(player, elo, games, most, seconds=0.05):
        """A match with Stockfish held to `elo`. Returns our scores, and says the result."""
        theirs, engine = stockfish(elo, seconds, engine_path)
        scores = match(player, theirs, games, generator, most)
        engine.quit()
        if sum(scores) in (0, games):  # lesson 39: one side took each point, no width to draw
            low, high = rate_interval([score == 1 for score in scores])
        else:
            low, high = interval([scores[i : i + 2] for i in range(0, games, 2)], generator)
        say(
            f"against Stockfish held to {elo:,}, {seconds} s a move: {sum(scores) / games:.2f} of "
            f"the points in {games} games, 95% interval {low:.2f} to {high:.2f}"
        )
        return scores

    # Lesson 41's chess model first, on the same yardstick. It reads 256 tokens, about 100 moves.
    before, old_words = load(cache / "chess_big")
    say("lesson 41's chess model:")
    floor = against(model_mover(before.to(device), old_words, generator)[0], 1320, 100, 100)

    model, words = load(cache / "strong")
    model, words.split = model.to(device), MOVES
    unseen = [line for month in sorted(cache.glob("strong_20*.val")) for line in month.open()]
    ratings = [int(line.split()[0][1:]) for line in unseen if line.strip()]
    assert ratings, "the tellings come from the unseen games: run python -m gpt.strong ids here"
    low = min(ratings) * 100
    top = max(r for r in set(ratings) if ratings.count(r) * 100 >= len(ratings)) * 100
    say(
        f"white's rating in the unseen games starts at {low:,}. The highest hundred that one "
        f"game in a hundred has is {top:,}"
    )

    # What should we tell it? Four ratings from the lowest to that one, each with a win, and
    # the last again with a draw. Each telling plays 100 games against Stockfish held to 1,900.
    steps = [low + round((top - low) * i / 300) * 100 for i in range(4)]
    tellings = {f"both players rated {r:,}, and its own side wins": (r, True) for r in steps}
    tellings[f"both players rated {top:,}, and a draw"] = (top, False)
    trial = {}
    for name, how in tellings.items():
        say(f"the stronger player, told: {name}")
        trial[name] = sum(against(best_legal(model, words, *how), 1900, 100, 240)) / 100
    name = max(trial, key=trial.get)
    ours, results = best_legal(model, words, *tellings[name]), []
    say(f"the stronger player in new games, told: {name}")
    for elo in (1400, 1600, 1800, 2000, 2200, 2400, 2600):
        results += [(elo, score) for score in against(ours, elo, 100, 240)]
    pairs = [results[i : i + 2] for i in range(0, len(results), 2)]  # an opening's two games
    best, again = rating(results), []
    for _ in range(200):  # lesson 24's bootstrap, over the pairs of games
        drawn = torch.randint(len(pairs), (len(pairs),), generator=generator).tolist()
        again.append(float(rating([game for i in drawn for game in pairs[i]])))
    low, high = (torch.tensor(again).quantile(q).item() for q in (0.025, 0.975))
    say(
        f"the stronger player is rated {best:,} on Stockfish's scale, "
        f"95% interval {low:,.0f} to {high:,.0f}"
    )
    # Stockfish's scale was set with far more time for a move. Does our 0.05 s flatter us?
    near = min((1400, 1600, 1800, 2000, 2200, 2400, 2600), key=lambda elo: abs(elo - best))
    slow = against(ours, near, 100, 240, seconds=0.5)
    save(
        "44_strong_play",
        torch=torch.__version__,
        device=device,
        floor=sum(floor) / len(floor),
        top=top,
        trial=trial,
        told=name,
        slow={"elo": near, "score": sum(slow) / len(slow)},
        results=results,
        rating={"best": best, "interval": [low, high]},
    )
