# 42. Serve on a GPU

By the end of this page vLLM serves the bigger story model that lesson 41 trained. You know how many requests a second it keeps up with, and what a million tokens cost, for our model and for one about 520 times its size.

Only our own engine can serve the bigger story model, and lesson 37 found that 0.58 ms of each 1.29 ms pass of that engine is the fixed cost of its calls.

## The idea

On the card our engine gives one listener 110 tokens a second, where lesson 37's laptop gave a smaller model 775. A pass takes 9.1 ms. A profiler, which records each job the card is handed and how long the card works on it, counts 634 jobs in one pass and 1.15 ms of work. The card idles for the other 8 ms.

vLLM is an open engine for serving models on graphics cards. It loads lesson 37's exported folder, given two more entries in its config file. Its log shows that at start-up it compiles the model's pass and records the pass's jobs once, as a graph it replays as one job. Like lesson 32's engine, it shares each pass among the requests it holds.

First we check that it computes what we compute, on the tokens both write and on the log chance each gives 200 unseen tokens.

Then we find where it stops keeping up: the script sends lesson 31's arrivals over the web and doubles the rate until the answers take over 1.1 times as long to leave as their requests took to arrive.

Last, we test whether memory is the limit. Lesson 31's sum divides the bytes the machine reads in a second by the bytes a token must read. On the laptop it allowed 4,231 tokens a second and one listener got 539. A big model spends its pass reading weights and should come close, so the script also serves Qwen2.5-7B-Instruct, an open model of 7.6 billion parameters, and scores it on 200 of our unseen stories.

That model also tests lesson 34's 8-bit weights, which made ours smaller and slower. vLLM's are floating-point numbers with one scale a grid, where lesson 34's were whole numbers with a scale a row.

## By hand

Lesson 31 timed a copy, which reads and writes. A pass only reads, so the script times a read: 741 GB a second on this card.

The big model's weights fill 15.2 GB at 16 bits. A token reads one row of the 1.1 GB grid of token rows, so that grid is left out. The scores layer is a grid of its own here, and vLLM keeps it at 16 bits.

| Weights | A token reads | Tokens a second memory allows |
|---|---|---|
| 16 bits | 15.2 − 1.1 = 14.1 GB | 741 / 14.14 = 52 |
| 8 bits | (15.2 − 2 × 1.1) / 2 + 1.1 = 7.6 GB | 741 / 7.6 = 97 |

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

The end token is the last id (lesson 5). The client, `ask`, sends token ids and reads them back from lesson 36's stream of events, so vLLM needs no tokenizer of ours. `load` gives each of lesson 31's requests a thread that sleeps until its moment, then asks.

## The proof

```bash
uv run pytest tests/test_production.py
```

- The config file gains the two entries and keeps lesson 37's.
- `ask` returns each token with the moment it came, and `load` sends each request at its moment.
- `scores` picks each prompt token's own log chance out of a reply that also holds the top token's.

Predict before you run: at 8 bits a token of the big model reads 7.6 GB where it read 14.1. Will one listener get twice the speed?

These run anywhere, against a stand-in server. The run needs lesson 41's `story_big` folder and an L40S card for under an hour. `start.sh` prints the machine's address, the `<host>` below. vLLM needs its own PyTorch, so it gets its own Python environment:

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/start.sh g6e.xlarge 90
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
the big model, 16 bits: a token reads 14.1 GB, so memory allows 52 tokens a second. One listener got 50
the big model, 8 bits: a token reads 7.6 GB, so memory allows 97 tokens a second. One listener got 73
200 unseen stories: our model 0.426 bits per byte, the big model 0.514, and at 8 bits +0.0095 nats a token (95% interval +0.0075 to +0.0115)
a million tokens written, our model: $0.065
a million tokens written, the big model, 16 bits: $0.155
```

Timings are from one rented machine and move from run to run.

vLLM computes what we compute to within 0.007. Our own code differs from itself by 0.0038 when we let the card use its faster 32-bit multiply, which keeps fewer digits. After 15 of 16 prompts the two write the same 128 tokens, and the 16th answer leaves ours where our two best choices were 0.0010 apart, less than either gap.

One listener gets 786 tokens a second where our engine gave 110. Memory would allow 12,732: for a model this small, fixed costs are still the limit.

vLLM keeps up at 128 requests a second, where a million tokens cost $0.065: 2.1 minutes of the card where lesson 36's server took 8 of the laptop. At 256 our clock averages 1,416 ms to the first token and the server's 83 ms. The server's clock starts when it begins work on a request, so the wait comes before that: in the server's line, or in our 2,560 sending threads, which share four processor cores with it. We did not find out which.

The big model meets the sum: one listener gets 50 tokens a second where memory allows 52. At 8 bits it gets 73 where memory allows 97: the reading falls to 0.54 of what it was, and by subtraction the rest of a pass grows from 1.1 ms to 3.5. So 8 bits buy 46% more speed for 0.0095 nats a token, where lesson 34's cost ours 0.0001.

On the 200 stories, each scored from its own start, our model gets 0.426 bits per byte (lesson 41's 0.467 is over all the unseen text) and the big model 0.514. A model about 520 times smaller, trained for $2.74 on such stories, predicts them better.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/stop.sh
```

`stop.sh` destroys each machine and lists any of ours left. Lessons 41 and 42 cost us $12.40 by the price list: `results/42_bill.json` has each machine's hours.

## What it still cannot do

Two things this course started are undone on the card: the bigger story model has not been taught to take an order, and lesson 36's own server has not run there. This was one machine for an afternoon. [MAP.md](../MAP.md) says where to read first for what comes after, such as copies of a model behind a router that shares requests among them.

You began with a loop that counted characters. One file of model code has since written stories and played chess, on a laptop and on a rented card.

## Check yourself

<details><summary>Prediction</summary>

No: 73 tokens a second against 50. The rest of a pass grows as the reading shrinks.
</details>

<details><summary>From lesson 37: why would smaller weights not speed up our own engine on the card?</summary>

The card works for 1.15 ms of a 9.1 ms pass. The rest is the fixed cost of 634 jobs, which smaller weights leave alone.
</details>

Go deeper:

- PyTorch, [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/). A pass recorded once and replayed as one job.
- PyTorch, [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html), on TensorFloat-32. The faster 32-bit multiply and the digits it drops.
