"""Lesson 42. Serve on a GPU: hand the model to vLLM on a rented card and measure the service."""

import http.client
import json
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager

from gpt.export import export, fastest


def handoff(folder, out):
    """Lesson 37's folder, and the two things vLLM asks for on top: the name its own code has
    for a model built like ours, and the id of the token that ends a text."""
    export(folder, out)
    config = json.loads((out / "config.json").read_text())
    config |= {"architectures": ["LlamaForCausalLM"], "eos_token_id": config["vocab_size"] - 1}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def post(port, path, **body):
    """Send one request to the server on this machine. Returns its reply, still open."""
    connection = http.client.HTTPConnection("127.0.0.1", port)
    connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    reply = connection.getresponse()
    assert reply.status == 200, reply.read()
    return reply


def ask(port, prompt, limit, clock=time.perf_counter, **how):
    """Ask the engine for `limit` tokens after a prompt, which is token ids or text.

    Returns the tokens it wrote and the moment each arrived. Like lesson 31's requests, it
    writes past the end token, so each answer is the same length.
    """
    reply = post(
        port,
        "/v1/completions",
        model="model",
        prompt=prompt,
        max_tokens=limit,
        stream=True,
        return_token_ids=True,
        ignore_eos=True,
        **how,
    )
    tokens, moments = [], []
    for line in reply:  # one line for each piece the engine sends
        if line.startswith(b"data: {"):
            new = json.loads(line[6:])["choices"][0].get("token_ids") or []
            tokens += new
            moments += [clock()] * len(new)
    return tokens, moments


def load(port, line):
    """Send lesson 31's line of arrivals to a server, each request at its moment and on a
    thread of its own. Returns lesson 31's records, for its `report`."""
    began, records = time.perf_counter(), []

    def one(at, request):
        time.sleep(max(0.0, at - (time.perf_counter() - began)))
        sent = time.perf_counter() - began  # a busy sender wakes late, and this says by how much
        moments = ask(port, request.prompt, request.limit, **request.how)[1]
        tokens = [moment - began for moment in moments]
        records.append({"arrived": at, "sent": sent, "tokens": tokens})

    threads = [threading.Thread(target=one, args=pair) for pair in line]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # A request that fails raises inside its own thread and leaves no record. Without this the
    # report would be worked out from the requests that got through.
    assert len(records) == len(threads), "a request failed: its error is printed above"
    return records


def scores(port, prompt):
    """The engine's log of the chance of each prompt token after the first, given those before."""
    reply = post(
        port,
        "/v1/completions",
        model="model",
        prompt=prompt,
        max_tokens=1,
        prompt_logprobs=1,
        return_token_ids=True,
    )
    choice = json.loads(reply.read())["choices"][0]
    pairs = zip(choice["prompt_token_ids"][1:], choice["prompt_logprobs"][1:])
    return [chances[str(token)]["logprob"] for token, chances in pairs]


@contextmanager
def serving(port, model, *options):
    """Start vLLM on a model, wait until it answers, and stop it afterwards."""
    command = ["vllm", "serve", str(model), "--served-model-name", "model"]
    command += ["--host", "127.0.0.1", "--port", str(port), *options]
    with open("vllm.log", "a") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:  # it loads the weights and warms up before it listens
                assert server.poll() is None, "vLLM stopped: read vllm.log"
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
                    break
                except OSError:
                    time.sleep(2)
            yield
        finally:
            server.terminate()
            server.wait()


if __name__ == "__main__" and sys.argv[1:] == ["late"]:
    # python -m gpt.production late: is the wait at the sweep's last rate the server's, or our
    # own sender's? The same arrivals again, from one sending process and then shared among four.
    import multiprocessing
    from pathlib import Path

    import torch

    from gpt.data import read
    from gpt.measure import arrivals, report
    from gpt.results import save, say

    cache, rows = Path("data/cache"), []
    text = read(cache / "stories_val.bin")
    prompts = [text[128 * i : 128 * i + 64].tolist() for i in range(3000)]
    with (
        serving(8000, cache / "story_vllm", "--skip-tokenizer-init", "--dtype", "float32"),
        multiprocessing.Pool(4) as pool,
    ):
        for prompt in prompts[:4]:  # warm the server up
            ask(8000, prompt, 64, temperature=0)
        pool.starmap(load, [(8000, [])] * 16)  # and the four processes
        for senders, rate in ((1, 128), (1, 256), (4, 256), (4, 512)):
            chance = torch.Generator().manual_seed(1)
            line = arrivals(prompts[: 10 * rate], rate, chance, temperature=0.8)
            shares = [(8000, line[i::senders]) for i in range(senders)]
            parts = [load(*shares[0])] if senders == 1 else pool.starmap(load, shares)
            records = [record for part in parts for record in part]
            first = sum(r["tokens"][0] - r["arrived"] for r in records) / len(records)
            late = sum(r["sent"] - r["arrived"] for r in records) / len(records)
            numbers = report(records) | {"first_token_mean": first, "sent_late_mean": late}
            rows.append({"rate": rate, "senders": senders} | numbers)
            say(
                f"{rate} requests a second from {senders} sending process"
                f"{'es' if senders > 1 else ''}: first token after {first * 1e3:,.0f} ms on "
                f"average, {numbers['tokens_a_second']:,.0f} tokens a second in all. "
                f"Each request was asked {late * 1e3:,.0f} ms late"
            )
    save("42_production_late", torch=torch.__version__, rows=rows)


if __name__ == "__main__" and not sys.argv[1:]:
    import math
    from pathlib import Path

    import torch

    from gpt.data import documents, read
    from gpt.engine import Engine, Request
    from gpt.evaluate import losses, paired_gap
    from gpt.generate import generate
    from gpt.gpu import PRICE
    from gpt.measure import arrivals, report
    from gpt.results import save, say
    from gpt.story import load as load_model

    assert torch.cuda.is_available(), "this lesson runs on a rented card: see aws/"
    PORT, BIG = 8000, "Qwen/Qwen2.5-7B-Instruct"
    cache = Path("data/cache")
    ours, words = load_model(cache / "story_big")
    ours.to("cuda")
    handoff(cache / "story_big", cache / "story_vllm")
    text = read(cache / "stories_val.bin")
    prompts = [text[128 * i : 128 * i + 64].tolist() for i in range(3000)]
    say(f"the card: {torch.cuda.get_device_name()}")

    def shared(count):
        """Lesson 32's engine writes 128 tokens for `count` listeners at once."""
        engine = Engine(ours, float("inf"), chunk=256)
        for prompt in prompts[:count]:
            engine.admit(Request(prompt, 128, how={"temperature": 0}))
        written = 0
        while engine.reading or engine.rows:
            written += len(engine.step())
        return written

    # Our own engine on the card, and where one pass of it spends its time.
    unseen = text[:201].to("cuda")  # lesson 24's `losses` scores each token after the first
    mine = [generate(ours, prompt, 128, temperature=0)[64:] for prompt in prompts[:16]]
    own = -losses(ours, unseen, 256, 256).cpu()
    torch.backends.cuda.matmul.allow_tf32 = True  # the card's faster 32-bit multiply (below)
    coarse = (-losses(ours, unseen, 256, 256).cpu() - own).abs().max().item()
    torch.backends.cuda.matmul.allow_tf32 = False
    alone = fastest(lambda: len(generate(ours, prompts[0], 128, temperature=0)) - 64)
    many = fastest(lambda: shared(64))
    kept = [{} for _ in ours.blocks]
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.no_grad():
        ours(torch.tensor([prompts[0]], device="cuda"), kept)
        with torch.profiler.profile(activities=activities) as watched:
            ours(torch.tensor([[prompts[1][0]]], device="cuda"), kept)
    kernels = [e for e in watched.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    busy = sum(e.device_time for e in kernels) / 1e6
    say(f"our engine: {alone:,.0f} tokens a second for one listener, {many:,.0f} for 64 at once")
    say(
        f"one token's pass takes {1e3 / alone:.1f} ms. Under the profiler the card works for "
        f"{busy * 1e3:.2f} ms of a pass, on {len(kernels)} jobs"
    )

    # Memory speed. Lesson 31 timed a copy, which reads each byte and writes it. A pass only
    # reads its weights, so here we time a read: the fastest of five sums over 4 GB on the card.
    block, reads = torch.ones(2**30, device="cuda"), []
    for _ in range(5):
        torch.cuda.synchronize()
        began = time.perf_counter()
        block.sum()
        torch.cuda.synchronize()
        reads.append(time.perf_counter() - began)
    moved = block.nbytes / min(reads)
    del block
    ours.to("cpu")
    torch.cuda.empty_cache()

    def counted():
        """Three of the server's own counts, from its metrics page: tokens written, first tokens
        sent, and the seconds those first tokens were waited for, added up."""
        page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics").read().decode()
        rows = [row.split() for row in page.splitlines() if row.startswith("vllm:")]
        names = ("generation_tokens_total", "time_to_first_token_seconds_count")
        names += ("time_to_first_token_seconds_sum",)
        return torch.tensor(
            [sum(float(row[-1]) for row in rows if row[0].startswith("vllm:" + n)) for n in names]
        )

    def one_listener(four):
        """Tokens a second and the wait for the first token, for one request alone. The first
        of four prompts warms the server up. Of the other three, the best of each number."""
        speeds, waits = [], []
        for prompt in four:
            began = time.perf_counter()
            moments = ask(PORT, prompt, 128, temperature=0)[1]
            speeds.append(127 / (moments[-1] - moments[0]))
            waits.append(moments[0] - began)
        return max(speeds[1:]), min(waits[1:])

    def sweep(name, asked, rate):
        """Double the rate of arrival until the server no longer keeps up: its answers end
        over a span a tenth longer than the span their requests arrived over."""
        rows = []
        while True:
            chance = torch.Generator().manual_seed(1)
            count = min(len(asked), max(100, round(10 * rate)))  # about ten seconds of arrivals
            before = counted()
            records = load(PORT, arrivals(asked[:count], rate, chance, temperature=0.8))
            numbers = report(records)
            tokens, firsts, waited = (counted() - before).tolist()
            came, went = ([r[key] for r in records] for key in ("arrived", "tokens"))
            ends = [moments[-1] for moments in went]
            numbers |= {
                "server_counted": tokens,
                "server_first_token_mean": waited / firsts,
                "first_token_mean": sum(m[0] - at for at, m in zip(came, went)) / len(came),
                "sent_late_mean": sum(r["sent"] - r["arrived"] for r in records) / len(records),
            }
            rows.append({"rate": rate} | numbers)
            say(
                f"{name}, {rate:,} requests a second: first token after "
                f"{numbers['first_token_p50'] * 1e3:,.0f} ms for half, "
                f"{numbers['first_token_p95'] * 1e3:,.0f} ms for 95 in 100, "
                f"{1 / numbers['seconds_a_token_p50']:,.0f} tokens a second each, "
                f"{numbers['tokens_a_second']:,.0f} in all"
            )
            if max(ends) - min(ends) > 1.1 * (max(came) - min(came)):
                return rows
            rate *= 2

    def dollars(rows):
        """Dollars for a million tokens written, at the last rate the server kept up with."""
        hour = PRICE[torch.cuda.get_device_name()][1]
        kept_up = rows[-2] if len(rows) > 1 else rows[0]  # a sweep that fell behind at once
        return hour / 3600 / kept_up["tokens_a_second"] * 1e6

    # vLLM on our model, in 32-bit numbers as ours are.
    with serving(PORT, cache / "story_vllm", "--skip-tokenizer-init", "--dtype", "float32"):
        theirs = [ask(PORT, prompt, 128, temperature=0)[0] for prompt in prompts[:16]]
        same = sum(a == b for a, b in zip(mine, theirs))
        gap = (own - torch.tensor(scores(PORT, unseen.tolist()))).abs().max().item()
        say(f"200 unseen tokens, largest gap between vLLM's log chances and ours: {gap:.6f}")
        say(f"the same gap between our code and itself with the faster multiply: {coarse:.6f}")
        say(f"prompts after which vLLM and our code write the same 128 tokens: {same} of 16")
        for prompt, a, b in zip(prompts, mine, theirs):
            if a != b:  # where an answer parts from ours, how close were our two best tokens?
                at = next(i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1])
                with torch.no_grad():
                    last = ours(torch.tensor([prompt + a[:at]]))[0, -1]
                best = torch.log_softmax(last, -1).topk(2).values
                say(
                    f"an answer parts from ours at its token {at + 1}, where the log chances "
                    f"of our two best tokens are {(best[0] - best[1]).item():.4f} apart"
                )
        speed, first = one_listener(prompts[:4])
        say(
            f"vLLM, one listener: first token after {first * 1e3:.0f} ms, {speed:,.0f} tokens a second"
        )
        small = sweep("vLLM", prompts, 16)
        sent = sum(row["served"] for row in small) * 64
        say(
            f"tokens written in the sweep: {sent:,} by our count, "
            f"{sum(row['server_counted'] for row in small):,.0f} by the server's"
        )
        for row in small[-2:]:  # the last rate the server kept up with, and the first it did not
            say(
                f"at {row['rate']} requests a second the first token took "
                f"{row['first_token_mean'] * 1e3:,.0f} ms on average by our clock, "
                f"{row['server_first_token_mean'] * 1e3:,.0f} ms by the server's"
            )

    # A model 500 times the size, at 16 bits and then at 8, on the same stories.
    stories = documents(text, words.end)[:200]
    told = [words.decode(doc[1:-1].tolist()) for doc in stories]
    asked = [words.decode(prompt) for prompt in prompts]
    # A stride of half the context: each token is scored with 128 tokens or more before it.
    ours_nats = [losses(ours, doc[:-1], 256, 128).tolist() for doc in stories]

    def bits(nats):
        """Bits per byte of the 200 stories, from each story's loss at each token."""
        return sum(map(sum, nats)) / sum(len(story.encode()) for story in told) / math.log(2)

    weights = 4 * sum(p.numel() for p in ours.parameters())
    say(f"this card reads {moved / 1e9:.0f} GB a second from its memory")
    say(
        f"our model reads {weights / 1e6:.0f} MB of weights for a token: at that speed, "
        f"{moved / weights:,.0f} tokens a second. vLLM gave one listener {speed:,.0f}"
    )
    big = {}
    for name, options in (("16 bits", []), ("8 bits", ["--quantization", "fp8_per_tensor"])):
        with serving(PORT, BIG, "--max-model-len", "2048", *options):
            nats = [[-x for x in scores(PORT, "<|endoftext|>" + story)] for story in told]
            fast, wait = one_listener(asked[:4])
            say(f"the big model, {name}, one listener: {fast:,.0f} tokens a second")
            rows = sweep(f"the big model, {name}", asked, 2)
        big[name] = {"nats": nats, "one_listener": fast, "first": wait, "rows": rows}
    # What a token reads of the big model. Not the grid of token rows: a token reads one row.
    # Its scores layer is a grid of its own, and vLLM leaves that one at 16 bits.
    folder = f".cache/huggingface/hub/models--{BIG.replace('/', '--')}/snapshots/*"
    folder = next(Path.home().glob(folder))
    sizes = json.loads((folder / "config.json").read_text())
    table = 2 * sizes["vocab_size"] * sizes["hidden_size"]
    heavy = sum(file.stat().st_size for file in folder.glob("*.safetensors"))
    reads = {"16 bits": heavy - table, "8 bits": (heavy - 2 * table) / 2 + table}
    for name, size in reads.items():
        say(
            f"the big model, {name}: a token reads {size / 1e9:.1f} GB, so memory allows "
            f"{moved / size:,.0f} tokens a second. One listener got {big[name]['one_listener']:,.0f}"
        )
    change, (low, high) = paired_gap(
        big["8 bits"]["nats"], big["16 bits"]["nats"], torch.Generator().manual_seed(0)
    )
    say(
        f"200 unseen stories: our model {bits(ours_nats):.3f} bits per byte, the big model "
        f"{bits(big['16 bits']['nats']):.3f}, and at 8 bits {change:+.4f} nats a token "
        f"(95% interval {low:+.4f} to {high:+.4f})"
    )
    costs = {"our model": dollars(small)}
    costs |= {f"the big model, {name}": dollars(run["rows"]) for name, run in big.items()}
    for name, cost in costs.items():
        say(f"a million tokens written, {name}: ${cost:.3f}")
    save(
        "42_production",
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        ours={
            "alone": alone,
            "together": many,
            "card_busy": busy,
            "jobs": len(kernels),
        },
        read_bytes_a_second=moved,
        vllm={
            "same": same,
            "gap": gap,
            "faster_multiply_gap": coarse,
            "one_listener": speed,
            "first": first,
            "rows": small,
        },
        big={name: {k: v for k, v in run.items() if k != "nats"} for name, run in big.items()},
        big_reads=reads,
        bits_per_byte={"ours": bits(ours_nats), "big": bits(big["16 bits"]["nats"])},
        eight_bit_gap={"nats": change, "interval": [low, high]},
        dollars_a_million_tokens=costs,
    )
