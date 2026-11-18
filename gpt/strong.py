"""Lesson 43. A stronger player: learn only from strong players' games, and measure the result
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


def strong(path, least):
    """The move lines of one file of Lichess games, for each game whose two players are both
    rated `least` or more. Bullet games, a minute or two a side, are left out."""
    wanted = (pc.field("WhiteElo") >= least) & (pc.field("BlackElo") >= least)
    wanted &= ~pc.match_substring(pc.field("Event"), "ullet")
    return pq.read_table(path, columns=["movetext"], filters=wanted)["movetext"].to_pylist()


# Lesson 5 cuts prose into runs of letters and runs of digits, so `Nf3` can never be one token.
# Here a chunk is a whole move with the space before it, and a common move becomes one token.
MOVES = re.compile(r" ?\S+|\s+")


def fused(q, k, v, mask=None):
    """Lesson 6's `attend`, done by PyTorch's own routine, which never keeps the grid of scores:
    less memory and less time for the same numbers. It hands back no weights."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None


def best_legal(model, words):
    """A player for lesson 39's games. Of the legal moves that are one token, it plays the one
    the model gives the highest chance, so the rules never have to pick a move for it."""
    import torch

    def move(board, played):
        ids = [words.end, *words.encode("".join(f" {one}" for one in played))]
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
                for game in games(strong(out.with_suffix(".parquet"), 2200)):
                    file.write(game + "\n")
                    kept += 1
        out.with_suffix(".parquet").unlink()
        out.with_suffix(".part").rename(out)  # a month cut short leaves no file by this name
        print(f"{month}: {kept:,} games between players rated 2,200 or more", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["ids"]:
    # python -m gpt.strong ids: the strong games as token ids, a file for each month. With
    # months named after it, only those: several of these can then run side by side.
    import json
    from pathlib import Path

    from gpt.bpe import train
    from gpt.data import held_out, tokenizer
    from gpt.gpu import more

    cache = Path("data/cache")
    months = [cache / f"strong_{month}.txt" for month in sys.argv[2:]]
    months = months or sorted(cache.glob("strong_20*.txt"))
    if not (cache / "strong_vocab.json").exists():  # lesson 5's trainer, on 50,000 games
        sample = months[0].read_text().split("\n")[:50_000]
        vocab = train(" " + " ".join(sample), 8192, MOVES)
        (cache / "strong_vocab.json").write_text(json.dumps([token.hex() for token in vocab]))
    words = tokenizer(cache / "strong_vocab.json")
    words.split = MOVES
    for month in months:  # lesson 18's steps: drop repeats, keep the unseen games out
        docs = [game for game in month.read_text().split("\n") if game]
        counts = more(month.with_suffix(".bin"), docs, words, tidy=lambda game: game, lead=" ")
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
    moves = sum(len(game.split()) for game in unseen)
    say(
        f"games between players rated 2,200 or more: {sum(c['read'] for c in counts):,} read, "
        f"{sum(c['written'] for c in counts):,} written for training, {len(ids):,} tokens"
    )
    say(
        f"tokens in the vocabulary: {words.end + 1:,}. Unseen games: "
        f"{(len(val_ids) - len(unseen)) / moves:.2f} tokens a move"
    )

    # The same step with lesson 6's attention and with PyTorch's fused routine in its place.
    size = shape(768, 12) | {"vocab": words.end + 1}
    torch.manual_seed(0)
    model = GPT(**size).to("cuda")
    ours = step_seconds(model, ids, half=True)
    gpt.model.attend = fused
    fast = step_seconds(model, ids, half=True)
    work = 3 * forward_flops(model, 32 * model.context, model.context)
    say(
        f"a training step of {sum(p.numel() for p in model.parameters()):,} parameters: "
        f"{ours * 1e3:.0f} ms with lesson 6's attention, {fast * 1e3:.0f} ms with the fused one"
    )
    del model

    budget = hours * 3600 * work / fast
    model, notes, steps, _ = long_run(
        cache, "strong", size, ids, val_ids, budget=budget, check=budget / 100, half=True
    )
    nats = evaluate(model, val_ids)
    for point in notes["curve"][:: max(1, len(notes["curve"]) // 8)]:
        say(f"step {point['step']}: train {point['train']:.3f}, unseen {point['val']:.3f} nats")
    card = torch.cuda.get_device_name(0)
    say(
        f"{steps:,} steps, {steps * 32 * size['context'] / len(ids):.1f} times through the games: "
        f"{nats:.3f} nats a token on unseen games, {hours:.0f} hours of one card, "
        f"${hours * PRICE[card][1]:.2f}"
    )
    out = cache / "strong"
    out.mkdir(exist_ok=True)
    save_llama(model.to("cpu"), out / "model.safetensors")
    (out / "vocab.json").write_text((cache / "strong_vocab.json").read_text())
    (out / "shape.json").write_text(json.dumps(size))
    save(
        "43_strong",
        torch=torch.__version__,
        device=card,
        counts=counts,
        seconds_a_step={"ours": ours, "fused": fast},
        steps=steps,
        val_nats=nats,
        curve=notes["curve"],
    )


if __name__ == "__main__" and sys.argv[1:2] == ["play"]:
    # python -m gpt.strong play: the stronger player against Stockfish held to five ratings.
    import shutil
    from pathlib import Path

    import torch

    import gpt.model
    from gpt.evaluate import interval
    from gpt.play import match, model_mover
    from gpt.results import save, say
    from gpt.story import load

    engine_path = shutil.which("stockfish") or "/usr/games/stockfish"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, generator = Path("data/cache"), torch.Generator().manual_seed(0)
    gpt.model.attend = fused

    def against(player, elo, games, most):
        """A match with Stockfish held to `elo`. Returns our scores, and says the result."""
        theirs, engine = stockfish(elo, path=engine_path)
        scores = match(player, theirs, games, generator, most)
        engine.quit()
        low, high = interval([scores[i : i + 2] for i in range(0, games, 2)], generator)
        say(
            f"against Stockfish held to {elo:,}: {sum(scores) / games:.2f} of the points in "
            f"{games} games, 95% interval {low:.2f} to {high:.2f}"
        )
        return scores

    # Lesson 41's chess model first, on the same yardstick. It reads 256 tokens, about 100 moves.
    before, old_words = load(cache / "chess_big")
    say("lesson 41's chess model:")
    floor = against(model_mover(before.to(device), old_words, generator)[0], 1320, 100, 100)

    model, words = load(cache / "strong")
    words.split = MOVES
    ours, results = best_legal(model.to(device), words), []
    say("the stronger player:")
    for elo in (1400, 1600, 1800, 2000, 2200):
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
    save(
        "43_strong_play",
        torch=torch.__version__,
        device=device,
        floor=sum(floor) / len(floor),
        results=results,
        rating={"best": best, "interval": [low, high]},
    )
