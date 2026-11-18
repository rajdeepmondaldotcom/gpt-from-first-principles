# 42. Serve on a GPU

By the end of this page vLLM, an engine built for rented cards, serves the storyteller that lesson 41 trained. You know the rate of arrival at which it stops keeping up, and what a million tokens cost, for our model and for one 500 times its size.

Nothing serves the bigger storyteller yet. The one engine we have is our own, and lesson 37 found that 0.58 of each 1.29 ms pass of it is the fixed cost of its calls.

## The idea

On the card our engine gives one listener 110 tokens a second. Each call of a pass now also hands the card a job and waits. A profiler, which records each job and how long the card works on it, counts 634 jobs in a 9.1 ms pass, with the card at work for 1.15 ms.

vLLM is an open engine for serving models on cards. It loads lesson 37's exported folder, with two more entries in the config file. Its log says that as it starts it compiles the model's pass and captures the card's jobs as one graph, which it replays. Like lesson 32's engine, it shares each pass among the requests inside.

We ask it three things.

Does it compute what we compute? Both get the same prompts, and we compare the tokens they write and the log of the chance each gives 200 unseen tokens.

When does it stop keeping up? Lesson 31's line of arrivals goes to the server over HTTP, and the script doubles the rate until answers leave over a span a tenth longer than their requests arrived over.

Is memory the limit yet? Lesson 31's sum divides the bytes a token must read by the bytes the machine reads in a second. On the laptop it allowed 4,231 tokens a second and one listener got 539. A big model should come close to the sum, so the script also serves Qwen2.5-7B-Instruct, an open model of 7.6 billion parameters, and scores it on 200 of our unseen stories.

That model also settles lesson 34, where 8-bit weights made ours smaller and no faster. If reading the weights is the limit, half the bytes should buy speed.

## By hand

The memory sum for the big model. Lesson 31 timed a copy, which reads each byte and writes it. A pass only reads its weights, so the script times a read: 741 GB a second on this card.

| Weights | A token reads | Tokens a second memory allows | One listener got |
|---|---|---|---|
| 16 bits | 14.1 GB | 741 / 14.1 = 52 | 50 |
| 8 bits | 7.6 GB | 741 / 7.6 = 97 | 73 |

The grid of token rows is not counted: a token reads one row of it. vLLM puts the four grids of each block in 8 bits and leaves the scores layer at 16.

## The code

```python
# gpt/production.py
def handoff(folder, out):
    """Lesson 37's folder, and the two things vLLM asks for on top: the name its own code has
    for a model built like ours, and the id of the token that ends a text."""
    export(folder, out)
    config = json.loads((out / "config.json").read_text())
    config |= {"architectures": ["LlamaForCausalLM"], "eos_token_id": config["vocab_size"] - 1}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
```

The client sends token ids and reads token ids, so vLLM needs no tokenizer of ours. The answer comes as lesson 36's stream, a line that starts with `data:` for each piece:

```python
# gpt/production.py
    tokens, moments = [], []
    for line in reply:  # one line for each piece the engine sends
        if line.startswith(b"data: {"):
            new = json.loads(line[6:])["choices"][0].get("token_ids") or []
            tokens += new
            moments += [clock()] * len(new)
    return tokens, moments
```

`load` gives each of lesson 31's requests a thread that sleeps until its moment and then asks, and keeps the moments as lesson 31's records, so its `report` reads them.

## The proof

```bash
uv run pytest tests/test_production.py
```

- The config file gains the two entries and keeps lesson 37's.
- `ask` asks for a stream of token ids and returns each token with the moment it came.
- `scores` takes each prompt token's own log chance, where the reply also holds the top token's.

These run anywhere, against a stand-in server that speaks vLLM's format. vLLM pins its own PyTorch, so on the rented machine it gets a Python environment of its own:

```bash
aws/run.sh <host> 'python -m venv ~/vllm && ~/vllm/bin/pip install -q vllm'
VENV='~/vllm' aws/run.sh <host> python -m gpt.production
```

```
the card: NVIDIA L40S
our engine: 110 tokens a second for one listener, 4,137 for 64 at once
one token's pass takes 9.1 ms. Under the profiler the card works for 1.15 ms of a pass, on 634 jobs
200 unseen tokens, largest gap between vLLM's log chances and ours: 0.006597
the same gap between our code and itself with the faster multiply: 0.003846
prompts after which vLLM and our code write the same 128 tokens: 15 of 16
an answer parts from ours at its token 41, where the log chances of our two best tokens are 0.0010 apart
vLLM, one listener: first token after 8 ms, 786 tokens a second
vLLM, 128 requests a second: first token after 24 ms for half, 47 ms for 95 in 100, 310 tokens a second each, 7,919 in all
vLLM, 256 requests a second: first token after 1,348 ms for half, 2,890 ms for 95 in 100, 135 tokens a second each, 12,527 in all
tokens written in the sweep: 317,440 by our count, 317,440 by the server's
at 128 requests a second the first token took 27 ms on average by our clock, 21 ms by the server's
at 256 requests a second the first token took 1,416 ms on average by our clock, 83 ms by the server's
this card reads 741 GB a second from its memory
our model reads 58 MB of weights for a token: at that speed, 12,732 tokens a second. vLLM gave one listener 786
the big model, 16 bits, one listener: 50 tokens a second
the big model, 16 bits, 64 requests a second: first token after 103 ms for half, 133 ms for 95 in 100, 28 tokens a second each, 3,341 in all
the big model, 16 bits, 128 requests a second: first token after 2,485 ms for half, 5,293 ms for 95 in 100, 20 tokens a second each, 4,699 in all
the big model, 8 bits, one listener: 73 tokens a second
the big model, 8 bits, 64 requests a second: first token after 63 ms for half, 77 ms for 95 in 100, 49 tokens a second each, 3,546 in all
the big model, 16 bits: a token reads 14.1 GB, so memory allows 52 tokens a second. One listener got 50
the big model, 8 bits: a token reads 7.6 GB, so memory allows 97 tokens a second. One listener got 73
200 unseen stories: our model 0.426 bits per byte, the big model 0.514, and at 8 bits +0.0095 nats a token (95% interval +0.0075 to +0.0115)
a million tokens written, our model: $0.065
a million tokens written, the big model, 16 bits: $0.155
a million tokens written, the big model, 8 bits: $0.146
```

Timings are from one rented machine, and they move from run to run.

vLLM computes what we compute to the third decimal: the largest gap in a log chance is 0.0066, and our own code moves 0.0038 from itself when the card may use its faster, coarser 32-bit multiply. After 15 of 16 prompts the two write the same 128 tokens. The other parts where our two best choices were 0.0010 apart, less than the gap.

One listener gets 786 tokens a second from vLLM where our engine gave 110. Memory would allow 12,732, so for a model this small fixed costs are still the limit.

vLLM keeps up at 128 requests a second: half the first tokens come within 24 ms, and a million tokens cost $0.065. At 256 half wait 1.3 seconds, though the server's own clock still says 83 ms. That clock starts when the server begins work on a request, so the line forms before then, on a machine whose four processor cores also run our 2,560 sending threads. We did not separate the two.

The big model meets the sum: one listener gets 50 tokens a second where memory allows 52. At 8 bits it gets 73 where memory allows 97. The reading is halved and the rest of a pass is not. Lesson 34's idea pays here: 46% more speed for 0.0095 nats a token. With 64 requests a second the card writes 3,341 tokens a second in all, and there 8 bits adds 6%.

On the 200 stories our model scores 0.426 bits per byte and the big model 0.514. A model 500 times smaller, trained for $2.74 on stories of this kind, predicts them better.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/stop.sh
```

`stop.sh` destroys each machine and lists what is left of ours: nothing. Lessons 41 and 42 cost us $12.40 by the price list: 4.9 hours of the L40S machine ($9.07), half an hour of four L4 cards ($2.29), 0.9 hours of a cheaper card to build this page's script ($0.91), and 14 cents of disk.

## What it still cannot do

This is one machine in one place for one afternoon. We have not put copies of a model behind a router, added machines as listeners arrive, swapped a model while it serves, or kept anything up for a week. [MAP.md](../MAP.md) says where to read first for each, and [PRACTICE.md](../PRACTICE.md) has routines for what a page cannot teach.

## Check yourself

Predict before you run: a token of the big model reads 14.1 GB at 16 bits, and the card reads 741 GB a second. About how many tokens a second can one listener get?

<details><summary>Answer</summary>

At most 741 / 14.1 = 52, and the run gives 50.
</details>

<details><summary>From lesson 34: 8-bit weights made our model no faster, and they make this one 46% faster. What differs?</summary>

What a pass spends its time on. Our model's pass is mostly the fixed cost of its calls, which smaller weights do not touch. The big model's pass is mostly reading its weights, and 8 bits halve the reading.
</details>

Go deeper:

- Kwon and others, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023). The vLLM paper.
- PyTorch, [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/). A pass recorded once and replayed as one job.
