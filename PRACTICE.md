# Practice

The lessons teach how a GPT works. Some of what AI labs ask for comes only from practice. Each of the nine routines below names what it rests on.

On 17 November 2026 we read 11 job postings for training and inference roles at Anthropic, OpenAI and Google DeepMind. Ten were open that day. DeepMind's had closed, and we read a copy that a job site had kept. This is what the eleven ask for that the lessons do not give you.

| Asked for | Postings, of 11 | Routines |
|---|---|---|
| Distributed-systems work | 8 | 3 and 4 |
| Profiling, kernels or compilers for accelerators | 5 | 5 |
| Reading and debugging a large codebase you did not write | 2 | 6 |
| Written reports with numbers in them | 1 | 8 |

A posting counts only where its own words ask for the thing. The table at the end says which posting asks for what.

Anthropic's guidance for candidates says its live interviews are done without AI help unless the interviewer says otherwise. Routines 1 and 2 rest on that. Routines 7 and 9 rest on no posting, and each says why it is here. The sources are at the end.

## 1. Write code against a clock, with no help

Pick one function in `gpt/` that you have read. Close the file, turn off the assistant and the autocomplete, and set a timer for 25 minutes. Write the function and run its test. Do one a day. Keep a list of what stopped you: an API you had to look up, an off-by-one, a shape.

## 2. Rebuild the core from a blank file

Three files hold up the rest: attention (lessons 6 and 7), the tokenizer (lesson 5) and the training loop (lesson 12). Delete each one's contents, write it again, and let the tests grade you.

```bash
uv run pytest tests/test_attention.py
```

```bash
uv run pytest tests/test_bpe.py
```

```bash
uv run pytest tests/test_train.py
```

Time each rebuild. Stop when attention takes under 20 minutes and each of the other two under 45.

## 3. Design a system on one page

Write three one-page designs. Each page has the same five parts: the load, the sum for one machine, the number of machines, what fails first, and what you would measure in the first week.

- A service that answers 100 requests a second with a first token inside 200 ms. Use lesson 31's measures and lesson 32's batching to say how many machines it takes.
- A training run on 64 graphics chips. Use lesson 17's count of memory and arithmetic, and lesson 20's shared slopes, to say what each chip holds and what travels between them.
- A harness that scores a new model against the last one. Use lesson 24: the split, the leak check, the interval, the paired difference.

Then give the page to someone for ten minutes with one instruction, to find the number that is wrong, and fix what they find.

Then do four sums at the size labs work at, on paper, timed and out loud. NVIDIA lists its H100 card with 80 GB of memory, read at 3.35 TB a second, and 1,979 million million multiplies and adds a second at 16 bits "with sparsity", which counts on some weights being zero. NVIDIA's page for the L40S lists half as much without, so count half: 990.

1. A model of 70 billion parameters trains with AdamW. What does lesson 17's count say it holds, and what does that do to lesson 20's plan of a whole copy for each worker?
2. That model has 80 blocks and 8 key heads of 128 numbers. What does lesson 23's cache hold for 32 listeners at a context of 128,000 tokens, in 16 bits?
3. By lesson 31's sum, what is the most tokens a second one listener can get from a model of 30 billion parameters in 16 bits on one card?
4. How many days do 1,000 cards take to train the 70 billion parameters on a trillion tokens, if each reaches 40% of the 990?

<details><summary>Answers</summary>

1. 70 × 10⁹ × 16 bytes = 1,120 GB. No card holds a copy, so the weights, slopes and averages must themselves be split. [MAP.md](MAP.md) has two rows on that, under Training.
2. 2 × 80 × 8 × 128 × 2 = 327,680 bytes a token. Times 128,000 tokens and 32 listeners, that is 1.34 TB: the memory of 17 cards.
3. The weights are 60 GB. 3,350 / 60 = 56 tokens a second.
4. 6 × 7 × 10¹⁰ × 10¹² = 4.2 × 10²³ multiplies and adds, by lesson 17's count. The cards do 1,000 × 0.4 × 990 × 10¹² = 3.96 × 10¹⁷ a second. That is 1.06 million seconds: 12 days.
</details>

## 4. Break a distributed system, then mend it

Lesson 20's four workers assume that none of them fails. Kill one in the middle of a run, and the coordinator stops with an error. Change `gpt/workers.py` so that training goes on with the workers that are left, and add a test that kills one.

```bash
uv run pytest tests/test_workers.py
```

Then write down each choice you had to make: how long to wait for a worker, who tries again, what happens to slopes that arrive late, and whether a step on three shares is still the step you wanted.

## 5. Profile before you guess

Lesson 31 timed whole requests and lesson 37 counted calls. A profiler says where the time inside a pass goes. Before you run one, write down the three operations you expect at the top of its table. Then run these lines with `uv run python`:

```python
from torch.profiler import profile

from gpt.generate import generate
from gpt.model import GPT
from gpt.scale import shape

model = GPT(**shape(256, 6))
with profile() as seen:
    generate(model, [0], 64)
print(seen.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
```

Set the table beside your guess. Then pick the top row, find the line of `gpt/` that causes it, and say what it would take to make it cheaper.

On a graphics card the question changes: how many jobs the card is handed, and how long it works on them. Lesson 42's script counts both for one pass, in `gpt/production.py`. Read how, then say which lines of `gpt/` hand over the most jobs.

## 6. Read a codebase you did not write

Start with [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a small serving engine. Trace one request from the moment it arrives to its last token, and write down the file and line where each idea from lessons 32 and 33 lives: the batch, the cache, the memory budget. Then open [vLLM](https://github.com/vllm-project/vllm) and find the same three things, starting from its scheduler. Give it two hours and stop. You do not need to read it all.

## 7. Criticise a paper in one page

No posting asks for this by name. It is here because each lesson rests on sources, most of them papers. Take one from a lesson's "Go deeper" each week and write four short paragraphs: what it claims, what evidence it gives, which step is weakest, and the one experiment you would run to test that step.

## 8. Report each experiment in writing

When you change something in this repo and run it, write half a page before you move on: the question, the one thing you changed, the seeds, the difference with its standard error by lesson 14's rule, and what you now believe. A reader who has ten minutes should be able to check your conclusion against your numbers.

## 9. Find a planted bug

No posting asks for this by name. It is here because lesson 13 showed that a broken model can train with no error, and finding the break is a skill of its own.

```bash
uv run python -m gpt.plant
```

Start from a clean checkout: the last command below throws away any change of yours in `gpt/` and `results/`. One line of `gpt/` is now wrong, one of nine the script can pick. Each of the nine turns a test red, which is how we know each is a bug. Do not run the tests, and do not look at the change. Use lesson 13's checks, short runs and printed numbers, and give yourself 30 minutes. If each check passes, compare a number a lesson prints with the one you get. Write down what you saw first and what it ruled out. Then read the answer:

```bash
git diff gpt
```

Put the code and any results you rewrote back:

```bash
git checkout gpt results
```

## Sources

Anthropic, [guidance for candidates on AI use](https://www.anthropic.com/candidate-ai-guidance).

| Posting | Asks for |
|---|---|
| Anthropic, [Staff + Senior Software Engineer, Inference](https://job-boards.greenhouse.io/anthropic/jobs/5385998008) | Distributed systems |
| Anthropic, [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008) | All four |
| Anthropic, [Performance Engineer](https://job-boards.greenhouse.io/anthropic/jobs/4020350008) | Distributed systems, kernels |
| Anthropic, [Research Engineer, Pretraining](https://job-boards.greenhouse.io/anthropic/jobs/5119713008) | Distributed systems |
| OpenAI, [Research Engineer](https://jobs.ashbyhq.com/openai/240d459b-696d-43eb-8497-fab3e56ecd9b) | Distributed systems |
| OpenAI, [Training Performance Engineer](https://jobs.ashbyhq.com/openai/6eb386ac-9056-4795-aa79-a27e105faf5c) | Distributed systems, profiling |
| OpenAI, [Researcher, Training](https://jobs.ashbyhq.com/openai/97d3670c-e75a-4bb2-a235-171765f5f10e) | None of the four |
| OpenAI, [Software Engineer, Kernel Performance & AI Tooling](https://jobs.ashbyhq.com/openai/e9627fa6-ac76-4899-9a93-9251419e61a0) | Kernels and compilers |
| OpenAI, [Software Engineer, Developer Productivity - Inference Runtime](https://jobs.ashbyhq.com/openai/9d48e2e6-41a9-4a90-8b3b-6cc960e95c2f) | Distributed systems |
| OpenAI, [Research Engineer / Research Scientist- Post Training](https://jobs.ashbyhq.com/openai/1c516e9f-c97d-4a40-8529-9871dac615a5) | A large codebase |
| Google DeepMind, [Software Engineer, Model Inference](https://jobs.anitab.org/companies/google-24698/jobs/90197320-software-engineer-model-inference-deepmind). Posted in August 2026 and closed. This is a job site's copy | Distributed systems, profiling and kernels |

Postings change each week, and a link here will stop working when its posting closes. The counts are what these eleven said on 17 November 2026.
