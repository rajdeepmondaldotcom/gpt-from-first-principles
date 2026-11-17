"""Lesson 38. The same code, a different world: chess games as text, checked by the rules."""

import hashlib
import re
import urllib.request

import torch

from gpt.generate import generate

# One month of rated games from Lichess, which gives its game records to the public domain.
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2014-03.pgn.zst"
SHA256 = "03c387eded7d0218bee672052472624576772f130c98cd9c20e8ddeed0735843"
# What a line of moves holds besides moves: notes in braces, move numbers, marks, the result.
NOT_A_MOVE = re.compile(r"\{[^}]*\}|\d+\.+|[?!]+|1-0|0-1|1/2-1/2|\*")


def fetch(path):
    """Download the month, once, and check its bytes are the ones we used."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != SHA256:
            raise ValueError(
                f"{path} is not the file this lesson measured. Delete it and run again."
            )


def moves(line):
    """One game's line from a Lichess file, as its moves alone: `e4 e5 Nf3 Nc6`."""
    return " ".join(NOT_A_MOVE.sub(" ", line).split())


def games(lines, least=20):
    """Each game of at least `least` moves, counting both sides, as one line of text."""
    for line in lines:
        if line.startswith("1. "):  # the one line of a game's record that holds its moves
            text = moves(line)
            if text.count(" ") + 1 >= least:
                yield text


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


def legal_moves(model, words, unseen, generator, longest=60):
    """Cut each unseen game at a random move and ask the model for the next one. For each game,
    whether the move it writes is legal."""
    out = []
    for game in unseen:
        played = game.split()
        cut = int(torch.randint(min(len(played), longest), (1,), generator=generator))
        out.append(legal(played[:cut], next_move(model, words, played[:cut])))
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path

    import zstandard  # reads the compressed file Lichess publishes

    from gpt.bpe import Tokenizer, train
    from gpt.data import held_out, prepare, read, tokenizer, write
    from gpt.evaluate import rate_interval
    from gpt.model import GPT
    from gpt.results import save, say
    from gpt.scale import shape
    from gpt.story import long_run, save_llama

    cache = Path("data/cache")
    if not (
        cache / "chess_counts.json"
    ).exists():  # lesson 18's steps, on games in place of stories
        fetch(cache / "lichess_2014-03.pgn.zst")
        with zstandard.open(cache / "lichess_2014-03.pgn.zst", "rt") as lines:
            docs = list(games(lines))
        kept, counts = prepare(docs, tidy=lambda game: game)  # `games` has done the cleaning
        unseen = [game for game in kept if held_out(game)]
        seen = [game for game in kept if not held_out(game)]
        vocab = train(" " + " ".join(seen[:15000]), 4096)  # about 5 MB of games
        words = Tokenizer(vocab)
        (cache / "chess_vocab.json").write_text(json.dumps([token.hex() for token in vocab]))
        for name, part in [("train", seen), ("val", unseen)]:
            ids = [i for game in part for i in (*words.encode(" " + game), words.end)]
            write(cache / f"chess_{name}.bin", ids)
            counts[name] = {"games": len(part), "tokens": len(ids)}
        (cache / "chess_val.txt").write_text("\n".join(unseen))
        (cache / "chess_counts.json").write_text(json.dumps(counts))
    counts = json.loads((cache / "chess_counts.json").read_text())
    words = tokenizer(cache / "chess_vocab.json")
    unseen = (cache / "chess_val.txt").read_text().split("\n")
    say(f"games read: {counts['read']:,}, repeats: {counts['repeats']:,}")
    say(f"near-repeats: {counts['near_repeats']:,}, games kept: {counts['kept']:,}")
    for name in ("train", "val"):
        part = counts[name]
        say(f"{name}: {part['games']:,} games, {part['tokens']:,} tokens")
    say(f"tokens in the vocabulary: {words.end + 1:,}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = read(cache / "chess_train.bin").to(device)
    val_ids = read(cache / "chess_val.bin").to(device)
    size = shape(256, 6) | {"vocab": words.end + 1}  # the story model, with chess's vocabulary
    torch.manual_seed(0)
    twin = GPT(**size)  # a copy on the processor, where writing one move at a time is faster

    def keep(out):
        """Save the twin as a model folder that lesson 21's `load` reads."""
        out.mkdir(exist_ok=True)
        save_llama(twin, out / "model.safetensors")
        (out / "vocab.json").write_text((cache / "chess_vocab.json").read_text())
        (out / "shape.json").write_text(json.dumps(size))

    def watch(model, step):
        """The share of 200 unseen positions the model continues with a legal move."""
        twin.load_state_dict(model.state_dict())
        hits = legal_moves(twin, words, unseen[:200], torch.Generator().manual_seed(0))
        print(f"  legal moves: {sum(hits)} of 200", flush=True)
        if step == 2500:  # a younger self, for lesson 39 to play against
            keep(cache / "chess_early")
        return {"legal": sum(hits) / 200}

    before = watch(twin, 0)["legal"]
    model, notes, steps, generator = long_run(cache, "chess", size, ids, val_ids, watch)
    count = sum(p.numel() for p in model.parameters())
    say(f"parameters: {count:,}, steps: {steps:,}, rate {notes['peak']:g}")
    say(f"before training: {before:.0%} of 200 moves are legal")
    for point in notes["curve"][:: max(1, len(notes["curve"]) // 9)]:
        say(
            f"step {point['step']}: unseen {point['val']:.3f} nats, "
            f"{point['legal']:.0%} of 200 moves are legal"
        )
    twin.load_state_dict(model.state_dict())
    hits = legal_moves(twin, words, unseen[:1000], generator)
    low, high = rate_interval(hits, generator)
    say(
        f"after training: {sum(hits)} of 1,000 moves are legal, 95% interval {low:.3f} to {high:.3f}"
    )
    game = generate(twin, [words.end], 120, stop=words.end, temperature=0.8, generator=generator)
    played = words.decode([i for i in game if i != words.end]).split()
    good = next((i for i in range(len(played)) if not legal(played[:i], played[i])), len(played))
    say(f"a game it writes alone: {' '.join(played)}")
    say(f"its first {good} moves are legal, of {len(played)}")

    keep(cache / "chess")
    save(
        "38_games",
        torch=torch.__version__,
        device=device,
        counts=counts,
        shape=size,
        steps=steps,
        peak=notes["peak"],
        curve=notes["curve"],
        legal_before=before,
        legal_after=sum(hits) / 1000,
        sample=" ".join(played),
    )
