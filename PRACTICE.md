# Practice

The lessons teach how a GPT works. Six skills that AI labs test come from practice and not from reading. Each one below has a routine you can start today.

On 17 November 2026 we read 11 job postings for training and inference roles at Anthropic, OpenAI and Google DeepMind. Six of the eleven ask for distributed-systems work. Five ask for profiling on accelerators. Three ask for reading and debugging a large codebase you did not write, and two for written reports with numbers in them. Anthropic's guidance for candidates says its live interviews are done without AI help unless the interviewer says otherwise. The sources are at the end.

## 1. Write code against a clock, with no help

Pick one function in `gpt/` that you have read. Close the file, turn off the assistant and the autocomplete, and set a timer for 25 minutes. Write the function and run its test. Do one a day. Keep a list of what stopped you: an API you had to look up, an off-by-one, a shape.

## 2. Rebuild the core from a blank file

Three files are worth knowing cold. Delete each one's contents, write it again, and let the tests grade you.

```bash
uv run pytest tests/test_attention.py
```

```bash
uv run pytest tests/test_bpe.py
```

```bash
uv run pytest tests/test_train.py
```

Attention is lessons 6 and 7, the tokenizer is lesson 5 and the training loop is lesson 12. Time each rebuild. Stop when all three take under an hour each.

## 3. Design a system on one page

Write three one-page designs, each with numbers.

- A service that answers 100 requests a second with a first token inside 200 ms. Use lesson 31's measures and lesson 32's batching to say how many machines it takes.
- A training run on 64 graphics chips. Use lesson 17's count of memory and arithmetic, and lesson 20's shared slopes, to say what each chip holds and what travels between them.
- A harness that scores a new model against the last one. Use lesson 24: the split, the leak check, the interval, the paired difference.

Then ask someone to attack the page for ten minutes, and fix what they find.

## 4. Read a codebase you did not write

Start with [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a small serving engine. Trace one request from the moment it arrives to its last token, and write down the file and line where each idea from lessons 32 and 33 lives: the batch, the cache, the memory budget. Then do the same in [vLLM](https://github.com/vllm-project/vllm), starting from its scheduler.

## 5. Criticise a paper in one page

Each lesson ends with papers under "Go deeper". Take one a week and write four short paragraphs: what it claims, what evidence it gives, which step is weakest, and the one experiment you would run to test that step.

## 6. Report each experiment in writing

When you change something in this repo and run it, write half a page before you move on: the question, the one thing you changed, the seeds, the difference with its standard error by lesson 14's rule, and what you now believe. A reader who has ten minutes should be able to check your conclusion against your numbers.

## Sources

- Anthropic, [guidance for candidates on AI use](https://www.anthropic.com/candidate-ai-guidance).
- Anthropic postings on [its job board](https://job-boards.greenhouse.io/anthropic): Software Engineer, Inference; Performance Engineer, Inference Systems; Performance Engineer; Research Engineer, Pretraining.
- OpenAI postings on [its job board](https://jobs.ashbyhq.com/openai): Research Engineer; Training Performance Engineer; Researcher, Training; Kernel Performance; Inference Runtime; Post-Training.
- Google DeepMind: Software Engineer, Model Inference, posted in August 2026 and since closed.

Postings change each week. The counts above are what those eleven said on that day.
