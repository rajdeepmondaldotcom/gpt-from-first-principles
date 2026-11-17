# Practice

The lessons teach how a GPT works. Some of what AI labs ask for comes only from practice. Each of the eight routines below names what it rests on.

On 17 November 2026 we read 11 job postings for training and inference roles at Anthropic, OpenAI and Google DeepMind. Ten were open that day. DeepMind's had closed, and we read a copy that a job site had kept. This is what the eleven ask for that the lessons do not give you.

| Asked for | Postings, of 11 | Routines |
|---|---|---|
| Distributed-systems work | 6 | 3 and 4 |
| Profiling, kernels or compilers for accelerators | 5 | 5 |
| Reading and debugging a large codebase you did not write | 3 | 6 |
| Written reports with numbers in them | 2 | 8 |

Anthropic's guidance for candidates says its live interviews are done without AI help unless the interviewer says otherwise. Routines 1 and 2 rest on that. Routine 7 rests on no posting, and says why it is here. The sources are at the end.

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

Time each rebuild. Stop when all three take under an hour each.

## 3. Design a system on one page

Write three one-page designs. Each page has the same five parts: the load, the sum for one machine, the number of machines, what fails first, and what you would measure in the first week.

- A service that answers 100 requests a second with a first token inside 200 ms. Use lesson 31's measures and lesson 32's batching to say how many machines it takes.
- A training run on 64 graphics chips. Use lesson 17's count of memory and arithmetic, and lesson 20's shared slopes, to say what each chip holds and what travels between them.
- A harness that scores a new model against the last one. Use lesson 24: the split, the leak check, the interval, the paired difference.

Then give the page to someone for ten minutes with one instruction, to find the number that is wrong, and fix what they find.

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

## 6. Read a codebase you did not write

Start with [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a small serving engine. Trace one request from the moment it arrives to its last token, and write down the file and line where each idea from lessons 32 and 33 lives: the batch, the cache, the memory budget. Then open [vLLM](https://github.com/vllm-project/vllm) and find the same three things, starting from its scheduler. Give it two hours and stop. You do not need to read it all.

## 7. Criticise a paper in one page

No posting asks for this by name. It is here because each lesson rests on sources, most of them papers. Take one from a lesson's "Go deeper" each week and write four short paragraphs: what it claims, what evidence it gives, which step is weakest, and the one experiment you would run to test that step.

## 8. Report each experiment in writing

When you change something in this repo and run it, write half a page before you move on: the question, the one thing you changed, the seeds, the difference with its standard error by lesson 14's rule, and what you now believe. A reader who has ten minutes should be able to check your conclusion against your numbers.

## Sources

- Anthropic, [guidance for candidates on AI use](https://www.anthropic.com/candidate-ai-guidance).
- Anthropic postings on [its job board](https://job-boards.greenhouse.io/anthropic): Software Engineer, Inference; Performance Engineer, Inference Systems; Performance Engineer; Research Engineer, Pretraining.
- OpenAI postings on [its job board](https://jobs.ashbyhq.com/openai): Research Engineer; Training Performance Engineer; Researcher, Training; Kernel Performance; Inference Runtime; Post-Training.
- Google DeepMind: Software Engineer, Model Inference. It was posted in August 2026 and had closed by 17 November.

Postings change each week. The counts above are what those eleven said.
