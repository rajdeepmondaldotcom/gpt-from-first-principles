"""Lesson 41. Train on a GPU: the same code on a rented card, and what an hour of it buys."""

import socket
import time
from pathlib import Path

import torch
from torch import distributed, multiprocessing

from gpt.cost import forward_flops
from gpt.data import clean, held_out, prepare, read
from gpt.gpt2 import batch, loss_on
from gpt.model import GPT
from gpt.optim import AdamW
from gpt.scale import predict, steps_for

# Dollars an hour in us-east-1, November 2026, for a machine with one card and with four:
# g6e.xlarge and g6e.12xlarge, g5.xlarge and g5.12xlarge, g6.xlarge and g6.12xlarge.
PRICE = {
    "NVIDIA L40S": {1: 1.861, 4: 10.493},
    "NVIDIA A10G": {1: 1.006, 4: 5.672},
    "NVIDIA L4": {1: 0.805, 4: 4.602},
}


def step_seconds(model, ids, size=32, repeats=10, half=False):
    """The fastest of `repeats` training steps on `size` windows, in seconds."""
    optimizer = AdamW(model.parameters(), rate=1e-4)
    generator, best = torch.Generator().manual_seed(0), float("inf")
    for count in range(repeats + 1):
        x, y = batch(ids, model.context, size, generator)
        began = time.perf_counter()
        model.zero_grad()
        with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=half):
            loss = loss_on(model, x, y)
        loss.backward()
        optimizer.step()
        loss.item()  # asking for the number waits for the card to finish
        if count:  # the first step sets the card up and is not a fair timing
            best = min(best, time.perf_counter() - began)
    return best


def more(path, docs, words, tidy=clean, lead=""):
    """Lesson 18's steps on a bigger pile: clean, drop repeats, keep the unseen documents out,
    and write the rest as token ids. Returns lesson 18's counts, and how many were written."""
    kept, counts = prepare(docs, tidy)
    seen = [doc for doc in kept if not held_out(doc)]
    with path.open("wb") as file:
        for at in range(0, len(seen), 20_000):  # a chunk at a time: the whole list would not fit
            part = seen[at : at + 20_000]
            ids = [i for doc in part for i in (*words.encode(lead + doc), words.end)]
            file.write(torch.tensor(ids, dtype=torch.int32).numpy().tobytes())
    return counts | {"written": len(seen)}


def pick(fitted, family, budget, most):
    """What lesson 19's law expects of each shape for `budget`, and the pick: the smallest shape
    within the law's miss on its own runs of the best, among those that read `most` tokens or
    fewer. Returns the rows and the pick."""
    rows = []
    for one in family:
        model = GPT(**one)
        steps = steps_for(model, budget)
        count, tokens = sum(p.numel() for p in model.parameters()), steps * 32 * one["context"]
        rows.append(
            {
                "shape": one,
                "parameters": count,
                "steps": steps,
                "tokens": tokens,
                "predicted": predict(fitted, count, tokens),
            }
        )
    fits = [row for row in rows if row["tokens"] <= most]
    best = min(row["predicted"] for row in fits)
    near = [row for row in fits if row["predicted"] - best <= fitted["miss"]]
    return rows, min(near, key=lambda row: row["parameters"])


def together(rank, cards, shape, path, steps, size, backend, half, port, out):
    """One of `cards` workers, each with a card of its own. Lesson 20's rule with no coordinator:
    the workers add their slopes up among themselves, and each steps with the average."""
    device = "cpu"
    if backend == "nccl":  # the cards' own link: this worker takes card number `rank`
        torch.cuda.set_device(rank)
        device = "cuda"
    distributed.init_process_group(
        backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=cards
    )
    ids = read(Path(path)).to(device)
    torch.manual_seed(0)  # the same seed: each worker starts from the same weights
    model = GPT(**shape).to(device)
    optimizer = AdamW(model.parameters(), rate=1e-3)
    generator, times = torch.Generator().manual_seed(0), []
    for _ in range(steps):
        began = time.perf_counter()
        x, y = batch(ids, model.context, size * cards, generator)  # each worker draws one batch
        model.zero_grad()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=half):
            loss = loss_on(model, x[rank::cards], y[rank::cards])  # and takes its own share
        loss.backward()
        for p in model.parameters():
            distributed.all_reduce(p.grad)  # now each worker holds the sum of everyone's slopes
            p.grad /= cards
        optimizer.step()
        if backend == "nccl":
            torch.cuda.synchronize()  # wait for the card to finish
        times.append(time.perf_counter() - began)
    if rank == 0:
        torch.save({name: value.cpu() for name, value in model.state_dict().items()}, out)
        Path(out).with_suffix(".txt").write_text(repr(min(times[2:])))  # the first steps set up
    distributed.destroy_process_group()


def race(cards, shape, path, steps, size, backend, out, half=False):
    """Run `steps` shared steps on `cards` workers, each with `size` windows a step, in 16-bit
    numbers if `half`. Returns the seconds the fastest step took and the weights they ended with."""
    with socket.socket() as probe:  # a free port for the workers to find each other on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(
            target=together, args=(rank, cards, shape, path, steps, size, backend, half, port, out)
        )
        for rank in range(cards)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not any(worker.exitcode for worker in workers), "a worker stopped with an error"
    return float(Path(out).with_suffix(".txt").read_text()), torch.load(out)


if __name__ == "__main__":
    import json
    import math
    import sys
    import urllib.request

    import zstandard

    from gpt.bpe import END
    from gpt.data import URL, tokenizer
    from gpt.evaluate import interval, rate_interval
    from gpt.games import cuts, fetch, games, legal_moves
    from gpt.generate import generate
    from gpt.gpt2 import evaluate
    from gpt.play import grabber, match, model_mover, random_mover, rate
    from gpt.results import save, say
    from gpt.scale import BUDGET, law, shape
    from gpt.story import CHECK, SHAPE, load, long_run, save_llama

    cache, cards = Path("data/cache"), torch.cuda.device_count()
    assert cards, "this script needs an NVIDIA card. Lesson 41's page says how to rent one."
    price = PRICE[torch.cuda.get_device_name(0)]

    if sys.argv[1:] == [
        "sizes"
    ]:  # after the run below: would a bigger shape have used the hour better?
        first = json.loads(Path("results/41_gpu.json").read_text())
        assert (cache / "stories_all.bin").exists(), "run python -m gpt.gpu on this machine first"
        ids, val_ids = (
            read(cache / name).to("cuda") for name in ("stories_all.bin", "stories_val.bin")
        )
        check, ended = first["budget"] * CHECK / BUDGET, []
        for row in first["rows"]:  # the pick, and each bigger shape the law was asked about
            count = row["parameters"]
            if row["shape"] == first["shape"]:
                nats = first["val_nats"]  # the run below trained this one
            elif row["steps"] < first["steps"]:  # a bigger shape: fewer steps for the budget
                torch.manual_seed(0)  # the start `long_run` makes, and its first check (lesson 13)
                start = evaluate(GPT(**row["shape"]).to("cuda"), val_ids[:20_000])
                even = math.log(row["shape"]["vocab"])
                if abs(start - even) >= 0.15:
                    say(
                        f"{count:>11,} parameters: not trained. Its first loss is {start:.3f} "
                        f"nats, {start - even:.3f} from ln {row['shape']['vocab']:,}, and lesson "
                        f"13's check allows 0.15"
                    )
                    continue
                model = long_run(
                    cache,
                    f"story_{count}",
                    row["shape"],
                    ids,
                    val_ids,
                    budget=first["budget"],
                    check=check,
                    half=True,
                )[0]
                nats = evaluate(model, val_ids)
            else:
                continue
            ended.append({"parameters": count, "nats": nats, "predicted": row["predicted"]})
            say(
                f"{count:>11,} parameters, {row['steps']:>7,} steps: {nats:.3f} nats on "
                f"unseen stories, where the law expected {row['predicted']:.3f}"
            )
            # Kept after each shape: the machine's time limit may come before the last one ends.
            (cache / "sizes_so_far.json").write_text(json.dumps(ended))
        save(
            "41_gpu_sizes",
            torch=torch.__version__,
            device=torch.cuda.get_device_name(0),
            sizes=ended,
        )
        raise SystemExit

    if cards > 1:  # the second machine: four cards, and lesson 20's question asked again
        assert cards in price, "add the hourly price of a machine with this many cards to PRICE"
        one = json.loads((cache / "story_big" / "shape.json").read_text())  # the one-card pick
        text = cache / "stories_train.bin"  # lesson 18's file: a timing needs no new text
        out = cache / "race.pt"
        # In 32-bit numbers, where rounding is small, on the batch of 32 windows the run above
        # trained with: do the cards end where one card does?
        single = race(1, one, text, 12, 32, "nccl", out)[1]
        several = race(cards, one, text, 12, 32 // cards, "nccl", out)[1]
        torch.manual_seed(0)  # the workers' seed, so these are the weights both runs began with

        def flat(weights):
            """Each weight in one row. The scores layer is the token rows again, so it is left out."""
            return torch.cat([weights[name].flatten() for name in single if name != "out.weight"])

        moved = (flat(single) - flat(GPT(**one).state_dict())).abs().mean().item()
        gaps = (flat(single) - flat(several)).abs()
        # In 16-bit numbers, as the run above trained: how long does a step take each way?
        alone = race(1, one, text, 12, 32 * cards, "nccl", out, half=True)[0]
        shared = race(cards, one, text, 12, 32, "nccl", out, half=True)[0]
        tokens = 32 * cards * one["context"]
        say(
            f"one card, {32 * cards} windows a step: {alone * 1e3:.0f} ms, {tokens / alone:,.0f} tokens a second"
        )
        say(
            f"{cards} cards, 32 windows each: {shared * 1e3:.0f} ms a step, "
            f"{tokens / shared:,.0f} tokens a second, {alone / shared:.2f} times as fast"
        )
        say(
            f"12 steps of 32 windows in 32-bit numbers: a weight moved {moved:.6f} on average. "
            f"One card's weights and {cards} cards' differ by {gaps.mean().item():.6f} on average, "
            f"{gaps.max().item():.6f} at most"
        )
        say(
            f"a million tokens of training: ${price[1] / 3600 / (tokens / alone) * 1e6:.4f} on "
            f"one card, ${price[cards] / 3600 / (tokens / shared) * 1e6:.4f} on {cards}"
        )
        save(
            "41_gpu_cards",
            torch=torch.__version__,
            device=torch.cuda.get_device_name(0),
            cards=cards,
            seconds_a_step={"one": alone, "many": shared},
            moved=moved,
            gap={"mean": gaps.mean().item(), "largest": gaps.max().item()},
        )
        raise SystemExit

    began, words = time.perf_counter(), tokenizer(cache / "stories_vocab.json")
    say(f"the card: {torch.cuda.get_device_name(0)}")

    # More text: the whole file that lesson 18 took the first 210 MB of.
    whole, text = cache / "tinystories_all.txt", cache / "stories_all.bin"
    if not text.exists():
        if not whole.exists():
            urllib.request.urlretrieve(URL, whole)
        counts = more(text, [doc for doc in whole.read_text().split(END) if doc.strip()], words)
        (cache / "stories_all.json").write_text(json.dumps(counts))
    counts = json.loads((cache / "stories_all.json").read_text())
    ids, val_ids = read(text).to("cuda"), read(cache / "stories_val.bin").to("cuda")
    say(
        f"stories read: {counts['read']:,}, kept: {counts['kept']:,}, "
        f"written for training: {counts['written']:,}, {len(ids):,} tokens"
    )

    # The laptop's model on this card: lesson 17's timing, taken again.
    torch.manual_seed(0)
    laptop = GPT(**SHAPE).to("cuda")
    mac = json.loads(Path("results/17_cost.json").read_text())["timings"]["mps"]
    plain, fast = step_seconds(laptop, ids), step_seconds(laptop, ids, half=True)
    work = 3 * forward_flops(laptop, 32 * laptop.context, laptop.context)
    say(
        f"the story model's training step: {mac['step'] * 1e3:.0f} ms on the laptop, "
        f"{plain * 1e3:.0f} ms on this card, {fast * 1e3:.0f} ms with 16-bit numbers"
    )
    say(
        f"multiplies and adds a second: {mac['flops_per_second'] / 1e9:,.0f} billion on the "
        f"laptop, {work / plain / 1e9:,.0f} billion here, {work / fast / 1e9:,.0f} with 16 bits"
    )

    # An hour of this card, measured on a model of middling size, and what the law would buy.
    middle = GPT(**shape(512, 8)).to("cuda")
    speed = 3 * forward_flops(middle, 32 * 256, 256) / step_seconds(middle, ids, half=True)
    budget = 3600 * speed
    say(f"an hour of this card: {budget:.1e} multiplies and adds, {budget / BUDGET:.0f} laptops")
    fitted = json.loads(Path("results/19_scale.json").read_text())
    story = json.loads(Path("results/21_story.json").read_text())
    done = [story["parameters"], story["steps"] * 32 * SHAPE["context"], story["val_nats"]]
    laws = {
        "lesson 19's law": fitted["final"],
        "the law with lesson 21's run added": law(fitted["runs"] + fitted["held_out"] + [done]),
    }
    family = [shape(256, 6), shape(384, 8), shape(512, 8), shape(640, 10), shape(768, 12)]
    rows, chosen = pick(fitted["final"], family, budget, most=1.5 * len(ids))
    for row in rows:
        say(
            f"{row['parameters']:>11,} parameters, {row['steps']:>7,} steps, "
            f"{row['tokens'] / len(ids):.1f} times through the text: predicted {row['predicted']:.3f} nats"
        )
    one = chosen["shape"]
    say(
        f"the pick: {chosen['parameters']:,} parameters, width {one['width']}, {one['layers']} blocks"
    )

    start = time.perf_counter()
    model, notes, steps, generator = long_run(
        cache,
        "story_big",
        one,
        ids,
        val_ids,
        budget=budget,
        check=budget * CHECK / BUDGET,
        half=True,
    )
    minutes = (time.perf_counter() - start) / 60
    sizes = json.loads(Path("results/18_data.json").read_text())["sizes"]["val"]
    nats = evaluate(model, val_ids)
    bits = nats * sizes["tokens"] / sizes["bytes"] / math.log(2)
    for point in notes["curve"][:: max(1, len(notes["curve"]) // 6)]:
        say(f"step {point['step']}: train {point['train']:.3f}, unseen {point['val']:.3f} nats")
    say(
        f"the bigger model, unseen stories: {nats:.3f} nats a token, {bits:.3f} bits per byte. "
        f"The laptop's model: {story['val_nats']:.3f} and {story['val_bits_per_byte']:.3f}"
    )
    guesses = {}
    for name, one_law in laws.items():
        guesses[name] = predict(one_law, chosen["parameters"], chosen["tokens"])
        say(f"{name} predicted {guesses[name]:.3f} nats: off by {nats - guesses[name]:+.3f}")
    say(f"the run: {minutes:.0f} minutes of one card, ${minutes / 60 * price[1]:.2f}")
    sample = generate(
        model, words.encode("Once upon a time"), 120, stop=words.end, temperature=0.8, top_p=0.95,
        generator=generator,
    )  # fmt: skip
    sample = words.decode([i for i in sample if i != words.end])
    say(f"sample: {sample!r}")
    out = cache / "story_big"
    out.mkdir(exist_ok=True)
    save_llama(model, out / "model.safetensors")
    (out / "vocab.json").write_text((cache / "stories_vocab.json").read_text())
    (out / "shape.json").write_text(json.dumps(one))

    # Chess: four months of games in place of one, the same shape, the same hour.
    chess_words = tokenizer(cache / "chess_vocab.json")
    text = cache / "chess_all.bin"
    if not text.exists():
        docs = []
        for month in ("2014-03", "2014-04", "2014-05", "2014-06"):
            path = cache / f"lichess_{month}.pgn.zst"
            if month == "2014-03":
                fetch(path)  # lesson 38's month, with its check
            elif not path.exists():
                urllib.request.urlretrieve(
                    f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst",
                    path,
                )
            with zstandard.open(path, "rt") as lines:
                docs += list(games(lines))
        counts = more(text, docs, chess_words, tidy=lambda game: game, lead=" ")
        (cache / "chess_all.json").write_text(json.dumps(counts))
    counts = json.loads((cache / "chess_all.json").read_text())
    ids, val_ids = read(text).to("cuda"), read(cache / "chess_val.bin").to("cuda")
    say(f"games written for training: {counts['written']:,}, {len(ids):,} tokens")
    size = one | {"vocab": chess_words.end + 1}
    start = time.perf_counter()
    model, notes, steps, generator = long_run(
        cache,
        "chess_big",
        size,
        ids,
        val_ids,
        budget=budget,
        check=budget * CHECK / BUDGET,
        half=True,
    )
    chess_minutes = (time.perf_counter() - start) / 60
    small, _ = load(cache / "chess")  # the laptop's chess model, from lesson 38
    twin = GPT(**size)  # on the processor, where writing one move at a time is faster
    twin.load_state_dict(model.state_dict())
    twin.eval()
    out = cache / "chess_big"
    out.mkdir(exist_ok=True)
    save_llama(twin, out / "model.safetensors")
    (out / "vocab.json").write_text((cache / "chess_vocab.json").read_text())
    (out / "shape.json").write_text(json.dumps(size))
    unseen = (cache / "chess_val.txt").read_text().split("\n")
    old = json.loads(Path("results/38_games.json").read_text())
    chess_nats = evaluate(model, val_ids)
    say(
        f"the bigger chess model: {sum(p.numel() for p in twin.parameters()):,} parameters, "
        f"{steps:,} steps, {steps * 32 * size['context'] / len(ids):.1f} times through the games, "
        f"unseen {chess_nats:.3f} nats where the laptop's had {old['curve'][-1]['val']:.3f}"
    )
    positions = cuts(unseen[:1000], torch.Generator().manual_seed(0))  # 1,000 unseen positions
    legal = {}
    for name, player in (("the laptop's model", small), ("the bigger model", twin)):
        hits = legal_moves(player, chess_words, positions)
        low, high = rate_interval(hits)
        legal[name] = sum(hits)
        say(f"{name}: {sum(hits)} of 1,000 moves are legal, 95% interval {low:.3f} to {high:.3f}")
    generator = torch.Generator().manual_seed(0)
    bigger, slips = model_mover(twin, chess_words, generator)
    others = {
        "the random mover": random_mover(generator),
        "the grabber": grabber(generator),
        "the laptop's model": model_mover(small, chess_words, generator)[0],
    }
    results, points = [], {}
    for name, other in others.items():
        scores = match(bigger, other, 200, generator)
        pairs = [scores[i : i + 2] for i in range(0, 200, 2)]
        if sum(scores) in (0, 200):  # one side took each point: the bootstrap has no width
            low, high = rate_interval([score == 1 for score in scores])
        else:
            low, high = interval(pairs, generator)
        points[name] = sum(scores) / 200
        say(
            f"the bigger model against {name}: {points[name]:.2f} of the points in 200 games, "
            f"95% interval {low:.2f} to {high:.2f}"
        )
        results += [[("the bigger model", name, score) for score in pair] for pair in pairs]
    say(f"the rules picked {slips['slips']:,} of its {slips['moves']:,} moves")
    # A rating on lesson 39's scale: its three results among the other players, 200 games each,
    # and one made-up draw between each two players, as there.
    before = json.loads(Path("results/39_play.json").read_text())["points"]
    names = [*others, "the bigger model"]
    known = [
        ("the grabber", "the random mover", before["the grabber against the random mover"]),
        ("the laptop's model", "the random mover", before["the model against the random mover"]),
        ("the laptop's model", "the grabber", before["the model against the grabber"]),
    ] * 200
    ties = [(a, b, 0.5) for i, a in enumerate(names) for b in names[i + 1 :]]
    ratings = rate([game for pair in results for game in pair] + known + ties, names)
    again = []
    for _ in range(100):  # lesson 24's bootstrap, over the pairs of games played here
        picks = torch.randint(len(results), (len(results),), generator=generator).tolist()
        again.append(rate([game for i in picks for game in results[i]] + known + ties, names))
    fitted_again = torch.tensor([one_fit["the bigger model"] for one_fit in again])
    low, high = fitted_again.quantile(0.025).item(), fitted_again.quantile(0.975).item()
    say(
        f"the bigger model: rated {ratings['the bigger model']:+.0f} with the random mover at 0, "
        f"95% interval {low:+.0f} to {high:+.0f}. The laptop's model: {ratings[names[2]]:+.0f}"
    )
    total = (time.perf_counter() - began) / 60
    say(f"the chess run: {chess_minutes:.0f} minutes, ${chess_minutes / 60 * price[1]:.2f}")
    say(f"this script, start to end: {total:.0f} minutes of one card, ${total / 60 * price[1]:.2f}")
    save(
        "41_gpu",
        torch=torch.__version__,
        device=torch.cuda.get_device_name(0),
        seconds_a_step={"laptop": mac["step"], "card": plain, "card_16_bit": fast},
        budget=budget,
        rows=rows,
        shape=one,
        steps=chosen["steps"],
        val_nats=nats,
        val_bits_per_byte=bits,
        predicted=guesses,
        minutes={"stories": minutes, "chess": chess_minutes, "all": total},
        sample=sample,
        chess={
            "nats": chess_nats,
            "legal": legal,
            "points": points,
            "slips": slips,
            "ratings": ratings,
        },
    )
