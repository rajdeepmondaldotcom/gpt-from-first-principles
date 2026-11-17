# The map

A topic is a lesson only if you cannot build, train, measure, adapt or serve a GPT without it, and only if we can prove it with a test: on a laptop for lessons 1 to 40, and on one rented machine for the last two. This page holds what that rule leaves out. Each topic gets one line on the problem it solves and the best open place to learn it. A resource can be in both tables, once for a topic and once as the best of its area.

Each link was fetched on 17 November 2026.

## What the lessons leave out

### Text and models

| Topic | The problem it solves | Learn it from |
|---|---|---|
| Other tokenizers | Lesson 5 builds byte pair encoding. WordPiece and unigram tokenizers cut text by other rules, and many models ship with them | Hugging Face, [the tokenizers chapter of its course](https://huggingface.co/learn/llm-course/chapter6/1) |
| Recurrent networks | Reading text one token at a time with a small memory. What language models were before attention | Karpathy, [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) |
| Other ways to mark position | Lesson 8 uses a table and lesson 15 rotation. Sine waves, a bias on distance and no marking at all are in use too | Weng, [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) |
| Mixture of experts | More weights without more arithmetic for each token: each token goes through a few of many MLPs | Hugging Face, [Mixture of Experts Explained](https://huggingface.co/blog/moe) |
| Images and sound | Feeding a transformer things that are not text, by cutting them into patches and treating each as a token | Dosovitskiy and others, [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) |

### Training

| Topic | The problem it solves | Learn it from |
|---|---|---|
| Muon | An optimizer for the grids inside the blocks, which its title calls hidden layers. The speed records of the next row use it | Jordan, [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) |
| What speeds training up | A record of each change that cut the time to train one small GPT, with the measurement for each | Jordan and others, [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| Training across many chips | Lesson 20 shares slopes. Real runs also split the model itself, across chips and across machines | Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) |
| Hardware arithmetic | Lesson 17's counting, for models a million times larger: where the time goes on real accelerators | Austin and others, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) |
| Tuning as a method | Lesson 14's discipline, as a full procedure for choosing each setting of a run | Godbole and others, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) |

### Using and measuring

| Topic | The problem it solves | Learn it from |
|---|---|---|
| Beam search | Looking for one high-chance text by keeping several partial texts alive | Hugging Face, [How to generate text](https://huggingface.co/blog/how-to-generate) |
| Classification | Using a language model to sort texts into classes | Raschka, [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch), chapter 6 |
| Calibration | Whether a model that says 70% is right 70% of the time | Guo and others, [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) |
| Models that mark models | Scoring answers that no exact check can score, with a second model as the marker | UK AI Security Institute, [Inspect: scorers](https://inspect.aisi.org.uk/scorers.html) |
| Standard tests | Running the public benchmarks the way papers run them | EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |

### Adapting

| Topic | The problem it solves | Learn it from |
|---|---|---|
| Retrieval | Answering from documents the model never trained on, by finding them and putting them in the prompt | Lewis and others, [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) |
| Agents and tools | Letting a model call programs and act on what they return | Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) |
| Constrained output | Making each answer fit a format, by allowing only the tokens that keep it valid | Willard and Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) |
| Reinforcement learning in depth | Lesson 27 has one method. The family around it, derived step by step | OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com/) |
| Post-training as a whole | Lessons 25 to 30 in one book, at the scale labs work at | Lambert, [the RLHF book](https://rlhfbook.com/) |

### Looking inside, and safety

| Topic | The problem it solves | Learn it from |
|---|---|---|
| Circuits | Working out which heads do what, by reading a small transformer's weights | Elhage and others, [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) |
| Sparse autoencoders | Finding directions in the stream that each mean one thing. One number of the stream, taken alone, takes part in many meanings | Bricken and others, [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) |
| Interpretability by hand | Lesson 40's probes and edits, as weeks of exercises | [ARENA](https://github.com/ARENA-education/ARENA_materials) |
| Red-teaming | Searching for prompts that make a model break its rules, and measuring how often they work | Mazeika and others, [HarmBench](https://arxiv.org/abs/2402.04249) |
| Safety tests as code | Lesson 30's counting, as a library with hundreds of ready tests | UK AI Security Institute, [Inspect](https://inspect.aisi.org.uk/) |

### Efficiency and serving

| Topic | The problem it solves | Learn it from |
|---|---|---|
| Pruning | Removing weights that matter least, where lesson 34 only stores them smaller | Han, [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940) |
| Rounding with care | Lesson 34 rounds each weight to the nearest step. Better rounding looks at what the layer computes | Frantar and others, [GPTQ](https://arxiv.org/abs/2210.17323) |
| Paged cache memory | Lesson 32 pads each request's cache to the longest in the batch. Engines keep it in small fixed pieces, so nothing goes on padding and requests that begin alike share pieces | Kwon and others, [the vLLM paper](https://arxiv.org/abs/2309.06180) |
| Fast attention on a graphics chip | Lesson 17's tiles, done so the grid of scores never leaves the chip's fastest memory | Dao and others, [FlashAttention](https://arxiv.org/abs/2205.14135) |
| Kernels and compilers | Why the same arithmetic can run fast or slow: code written for one kind of chip, and the programs that write such code | He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) |
| The words of the hardware | What each part of a graphics chip is called and does | Modal, [GPU Glossary](https://modal.com/gpu-glossary) |
| Many machines for one service | Routing requests across copies of a model, adding copies under load, and replacing them safely | Modular, [LLM Inference Handbook](https://github.com/modular/llm-inference-handbook), begun at BentoML |
| Staying up | What a service does when more arrives than it can serve, and how to say what "up" means | Google, [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/) |
| One model across chips | Splitting a single model's arithmetic over several chips to answer faster | Pope and others, [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) |
| Load tests | Lesson 36 uses `ab`. Tools built for language models sweep the rate and report each wait | [GuideLLM](https://github.com/vllm-project/guidellm) |
| Lines and waits | The mathematics of lesson 31's waiting line | Harchol-Balter, [Performance Modeling and Design of Computer Systems](https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html) |

## The best open resources, by area

Licences are as each source stated them on the day we checked. "None stated" means free to read with no licence given.

| Area | Resource | Licence | Why this one |
|---|---|---|---|
| The whole path, at university depth | [Stanford CS336](https://cs336.stanford.edu/) | Assignments MIT | Five assignments graded by tests: the model, systems, scaling, data, alignment |
| The whole path, small | Karpathy, [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) and [nanochat](https://github.com/karpathy/nanochat) | MIT | Videos from zero to a GPT, and one current recipe from tokenizer to chat |
| A GPT in one file | Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) | MIT | GPT-2's training, reproduced in a few hundred lines. Its author now points readers to nanochat |
| The mathematics underneath | Deisenroth and others, [Mathematics for Machine Learning](https://mml-book.github.io/) | None stated | Linear algebra, calculus and probability, written for this use |
| Deep learning as a textbook | Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | CC BY-NC-ND 4.0 | Losses as maximum likelihood, starting weights, why models generalise |
| Running experiments | [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | CC BY 4.0 | Choosing settings by experiment |
| Small-model training in practice | Hugging Face, [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) | None stated | A real training log, with each choice and its ablation |
| Cost and scale | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | MIT | Memory, arithmetic and time, counted for real accelerators |
| Data | Hugging Face, [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) and [datatrove](https://github.com/huggingface/datatrove) | Apache-2.0 | A real filtering and dedup pipeline, with each filter measured |
| Scaling laws | Hoffmann and others, [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | None stated | The law lesson 19 fits, at a budget 200 million times ours |
| Post-training | Lambert, [the RLHF book](https://rlhfbook.com/) | CC BY-NC-SA 4.0 | Reward models, learning from checks, preferences |
| Evaluation | Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) | None stated | Statistics for comparing models |
| Evaluation in code | [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | MIT | Tasks with checks, as labs run them |
| Interpretability | [ARENA](https://github.com/ARENA-education/ARENA_materials) and [Transformer Circuits](https://transformer-circuits.pub/) | None stated | Circuits, probes and features, by exercise and by paper |
| Efficient inference | [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940) | None stated | Quantization, pruning and distillation in depth |
| Serving engines | [vLLM](https://github.com/vllm-project/vllm) and [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Apache-2.0, MIT | The engine most services run, and one small enough to read |
| Serving in production | Modular, [LLM Inference Handbook](https://github.com/modular/llm-inference-handbook) | CC BY 4.0 | Operations: routing, scaling, cost, what to watch |
| Reliability | Google, [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/) | CC BY-NC-ND 4.0 | Overload, monitoring, and what a promise about uptime means |
| Apple hardware | [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | The engine lesson 37 hands our model to |
| Engineering lore | Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) | CC BY-SA 4.0 | Hardware, debugging and failures, in one engineer's notes |

## What a laptop cannot claim

Lessons 31 to 37 prove a correct, measured engine on one machine. They do not show hundreds of streams at once, waits under real contention, several copies of a model behind a router, a release rolled out and rolled back, or a cost read off a bill. Lessons 41 and 42 take the first steps of that on a rented machine. You learn the rest on the job, and the tables above list where to read first.
