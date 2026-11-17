# Glossary

Each term the lessons define, in one line, with the lesson that defines it. The lessons use plain words where they can. The first table sets each plain word beside the word a paper or an interviewer would use, so that you can read both.

## Our words and the field's

| Our word | The field's word |
|---|---|
| slope | partial derivative. All the slopes together are the gradient |
| grid | matrix |
| row of numbers | vector |
| score | logit |
| chance | probability |
| surprise | negative log-likelihood. Averaged over a text, the cross-entropy loss |
| place | position |
| table of places | learned positional embedding |
| rotary positions | RoPE |
| bend | activation function, or nonlinearity |
| spread | standard deviation |
| the stream | residual stream |
| path around a layer | residual connection, or skip connection |
| grouped heads | grouped-query attention, GQA |
| the cache | KV cache |
| unseen text | validation set, or held-out set |
| storing its text | memorisation, overfitting. The gap is the generalisation gap |
| multiplies and adds | FLOPs |
| the processor, the graphics chip | CPU, GPU. MPS is PyTorch's name for Apple's graphics chip |
| rate | learning rate |
| a task's check | verifier, or grader |
| pass at k | pass@k |
| order | instruction, or prompt |
| padded places left out of the loss | padding, and loss masking |
| the score (lesson 27) | reward |
| learning from a score | reinforcement learning. Lesson 27's rule is the policy gradient, or REINFORCE |
| what is subtracted from each score | baseline |
| keep what passes, and fine-tune on it | expert iteration, or filtered fine-tuning |
| the scorer | reward model |
| pairs where a reader liked one more | preference data, or pairwise comparisons |
| the bet's loss | Bradley-Terry loss, or pairwise ranking loss |
| trained hard against the scorer, the readers' number falls | reward hacking. The general rule is Goodhart's law |
| the reference | reference policy |
| a miss, an overreach | under-refusal, over-refusal. Also false negative, false positive |
| the wait before anything appears | time to first token, TTFT |
| the gap between tokens | inter-token latency, or time per output token |
| the line | queue |
| joining and leaving between two tokens | continuous batching |
| reading a prompt in chunks | chunked prefill |
| `real` | attention mask, or padding mask |
| taking a request only if memory allows | admission control |
| the store of beginnings | prefix cache. Services sell it as prompt caching |
| throwing out the piece used longest ago | least-recently-used eviction, LRU |
| a scale for each row | per-channel quantization |
| the draft | draft model |
| keep chance | acceptance probability |
| the leftover | residual distribution |
| copying the big model's chances | distillation, with soft targets |
| saying no when the line is full | load shedding |
| requests inside = arrival rate × time inside | Little's law |
| a run of small events | server-sent events, SSE |
| the fixed cost of each call | framework overhead |
| a move as text | SAN, standard algebraic notation. A file of games is PGN |
| rating | Elo rating |
| nudging the stream | intervention. Close relatives are activation patching and steering |
| reading an earlier stream through the final layer | logit lens |
| fitting the probe to shuffled boards | control task |

## The terms

- **95% interval**: The middle 95% of the means found by drawing again from the data 1,000 times. [Lesson 24](lessons/24-evaluation.md)
- **Adam**: Keeps running averages of each parameter's slope and squared slope, and steps by rate × average slope / square root of average squared slope. [Lesson 11](lessons/11-optimizers.md)
- **AdamW**: Adam plus weight decay, applied to weight grids and not to biases, gains or shifts. [Lesson 11](lessons/11-optimizers.md)
- **attention**: Each token's result is the weighted sum of values, with weights from softmax of its query's dot products with the keys. [Lesson 6](lessons/06-attention.md)
- **attention weights**: Softmax of a token's scores, summing to 1. Worked out afresh for each text and not parameters. [Lesson 6](lessons/06-attention.md)
- **autograd** (automatic differentiation): Code that works out the loss once, lets each number remember what it was made from, then walks backwards to get each slope. [Lesson 2](lessons/02-gradients.md)
- **baseline**: A number subtracted from each score before a step of learning from a score. Here it is the average score of a story's group. [Lesson 27](lessons/27-learning-from-a-score.md)
- **batch**: The random handful of pairs a step learns from, in place of all the text. [Lesson 3](lessons/03-tensors.md)
- **batching**: Giving the model a row for each reader, so that one pass writes a token for each of them. [Lesson 32](lessons/32-batching.md)
- **bend**: Squashing each number with a function such as tanh between linear layers, because two linear layers in a row do no more than one. [Lesson 4](lessons/04-a-neural-language-model.md)
- **beta**: In direct preference optimization, the small number that multiplies a pair's lead. It sets how large a lead counts as large. [Lesson 29](lessons/29-preferences-without-a-scorer.md)
- **bias**: A constant added to each weighted sum in a linear layer. [Lesson 4](lessons/04-a-neural-language-model.md)
- **bigram**: A model that looks one character back. [Lesson 1](lessons/01-predict-the-next-token.md)
- **bit**: Unit of surprise measured with logarithms to base 2. Bits equal nats divided by ln 2. [Lesson 1](lessons/01-predict-the-next-token.md)
- **bits per byte**: A text's whole loss in bits, divided by the bytes the text takes. The scoreboard for the whole course. [Lesson 1](lessons/01-predict-the-next-token.md)
- **block**: Attention with three parts added: an MLP, a path around each layer, and a normalization. [Lesson 9](lessons/09-the-block.md)
- **bootstrap**: Drawing again, with repeats allowed, as many items as you hold and taking the mean again, many times. [Lesson 24](lessons/24-evaluation.md)
- **broadcasting**: PyTorch stretching a column or smaller tensor across all places of a larger one in arithmetic. [Lesson 3](lessons/03-tensors.md)
- **byte**: A number from 0 to 255. A computer stores text as bytes. [Lesson 5](lessons/05-tokens.md)
- **byte pair encoding** (BPE): Make the most frequent pair of neighbouring tokens a new token, replace it everywhere, and repeat until the vocabulary is the size you want. [Lesson 5](lessons/05-tokens.md)
- **causal mask**: The rule that each token's result depends only on what came before it. [Lesson 7](lessons/07-causal-multi-head-self-attention.md)
- **chain rule**: Slopes multiply along a path from a number to the loss, and add when the number reaches the loss by two paths. [Lesson 2](lessons/02-gradients.md)
- **chance** (probability): How likely something is, as a number from 0 to 1. [Lesson 1](lessons/01-predict-the-next-token.md)
- **check**: Of a run, a fact that holds for a healthy one and fails for a broken one. [Lesson 13](lessons/13-is-it-working.md) Of a task, the rule that says whether an answer passes. [Lesson 24](lessons/24-evaluation.md)
- **checkpoint**: A file that lets a run stop and start: parameters, optimizer averages and step count, step number, and the random generator state. [Lesson 12](lessons/12-the-training-loop.md)
- **client**: A program that sends requests to a server. [Lesson 36](lessons/36-the-server.md)
- **clipping**: If the gradient's length is above the limit, scale the gradient down to the limit and keep its direction. [Lesson 12](lessons/12-the-training-loop.md)
- **comparison loss**: The surprise at a reader's choice between two stories, when the difference between their scores is a bet on that choice. [Lesson 28](lessons/28-a-model-of-what-people-prefer.md)
- **context**: The most tokens the model reads at once. [Lesson 8](lessons/08-position.md)
- **coordinator**: The process that draws the batch, sends each worker its share, averages the slopes and sends the average back. [Lesson 20](lessons/20-many-workers.md)
- **cosine and sine**: The two coordinates of the point (1, 0) after a turn by that angle. [Lesson 15](lessons/15-rotary-positions.md)
- **cross-entropy**: The loss: the model's average surprise over a text. [Lesson 1](lessons/01-predict-the-next-token.md)
- **decoding**: Turning token ids back into text. [Lesson 5](lessons/05-tokens.md)
- **device**: Where a tensor lives: cpu, the processor, or a graphics chip. [Lesson 10](lessons/10-the-gpt.md)
- **direct preference optimization** (DPO): Learning from pairs with no scorer. The loss rewards the better story for gaining on the worse one, each measured against a frozen copy of the model. [Lesson 29](lessons/29-preferences-without-a-scorer.md)
- **distillation**: Training a small model to copy a large model's whole row of chances, where plain training has one right token. [Lesson 35](lessons/35-speculative-decoding.md)
- **dot product**: Multiply two vectors entry by entry and add the products. Large when they have large numbers of the same sign in the same places. [Lesson 6](lessons/06-attention.md)
- **draft**: The small model that guesses tokens ahead in speculative decoding. [Lesson 35](lessons/35-speculative-decoding.md)
- **dropout**: In training, set each number a layer adds to the stream to 0 with a fixed chance, and divide the rest by the chance kept. [Lesson 13](lessons/13-is-it-working.md)
- **Elo rule**: The score to expect from two ratings: 1 / (1 + 10^((S - R) / 400)) for a player rated R against one rated S. [Lesson 39](lessons/39-can-it-play.md)
- **embedding**: The row of numbers a token is given. The rows start random and training moves them. [Lesson 4](lessons/04-a-neural-language-model.md)
- **encoding**: Turning text into token ids. [Lesson 5](lessons/05-tokens.md)
- **end token**: An extra token that marks where a document ends. It never comes out of ordinary text. [Lesson 5](lessons/05-tokens.md)
- **engine**: The code that takes requests in and writes their tokens. [Lesson 31](lessons/31-latency-and-throughput.md)
- **fine-tuning**: Training a trained model further on new text. [Lesson 25](lessons/25-fine-tuning.md)
- **fingerprint**: A short number worked out from a document's text, the same for the same text. [Lesson 18](lessons/18-data.md)
- **floor**: The loss that neither the model-size term nor the token-count term removes. [Lesson 19](lessons/19-how-big-should-it-be.md)
- **forward pass**: The pass from tokens to loss. [Lesson 12](lessons/12-the-training-loop.md)
- **gain**: How much more likely our model makes a story than the reference does: the difference of the two log chances. [Lesson 29](lessons/29-preferences-without-a-scorer.md)
- **gap**: The difference between the loss on training text and the loss on unseen text. [Lesson 13](lessons/13-is-it-working.md)
- **GELU**: A bend that gives about 0 for large negative numbers and about the number itself for large positive ones. [Lesson 9](lessons/09-the-block.md)
- **GPT**: A generative pre-trained transformer: it writes, it is trained once on a great deal of text, and its body is a stack of attention blocks. [Lesson 10](lessons/10-the-gpt.md)
- **GPU**: The graphics chip. [Lesson 12](lessons/12-the-training-loop.md)
- **gradient**: The slopes for all the parameters, taken together. [Lesson 2](lessons/02-gradients.md)
- **gradient descent**: Repeating the rule: new value = old value - rate × slope. [Lesson 2](lessons/02-gradients.md)
- **greedy**: The plainest picking rule: take the token with the top score each time, so the same prompt gives the same story. [Lesson 22](lessons/22-sampling.md)
- **grouped heads**: Several query heads read from one key head and one value head. [Lesson 16](lessons/16-todays-parts.md)
- **head**: One equal slice of the queries, keys and values, with attention run on it alone and weights of its own. [Lesson 7](lessons/07-causal-multi-head-self-attention.md)
- **hidden units**: The bent numbers between a model's two linear layers. [Lesson 4](lessons/04-a-neural-language-model.md)
- **keep chance**: In speculative decoding, the chance that a drafted token is kept: 1 where the big model gives it at least the draft's chance, and p / q otherwise. [Lesson 35](lessons/35-speculative-decoding.md)
- **key**: What a token offers, made from its embedding by a linear layer. [Lesson 6](lessons/06-attention.md)
- **KV cache**: The key and value of each token read so far, kept in each block so a new token is run alone. [Lesson 23](lessons/23-the-kv-cache.md)
- **language model**: A model that is given some text and says how likely each possible next piece is. [Lesson 1](lessons/01-predict-the-next-token.md)
- **latency**: The waits a reader feels: the time to the first token, and the gap between tokens. [Lesson 31](lessons/31-latency-and-throughput.md)
- **law**: loss = floor + a / parameters^alpha + b / tokens^beta, fitted to short runs to predict the loss of runs not seen. [Lesson 19](lessons/19-how-big-should-it-be.md)
- **layer normalization**: Subtract each token's vector's mean, divide by its spread, then apply a learned gain and shift. [Lesson 9](lessons/09-the-block.md)
- **lead**: The better story's gain minus the worse story's. [Lesson 29](lessons/29-preferences-without-a-scorer.md)
- **leak**: A piece of the unseen text that is also in the training text. [Lesson 24](lessons/24-evaluation.md)
- **learning rate**: The size of the step in the gradient descent rule. [Lesson 2](lessons/02-gradients.md)
- **least squares**: The exact answer for the weights that make the sum of squared misses smallest. [Lesson 19](lessons/19-how-big-should-it-be.md)
- **leftover**: What is drawn from after a drafted token is thrown out: the chance the big model has and the draft lacks, shared out to sum to 1. [Lesson 35](lessons/35-speculative-decoding.md)
- **legal move**: A move the rules of chess allow in the position at hand. [Lesson 38](lessons/38-the-same-code-a-different-world.md)
- **Llama**: A family of open models from Meta. [Lesson 16](lessons/16-todays-parts.md)
- **logarithm**: The natural logarithm of x, ln x, is the power that e, about 2.718, must be raised to in order to give x. [Lesson 1](lessons/01-predict-the-next-token.md)
- **look-alike**: A harmless word that resembles a listed one, used to count the orders a rule refuses wrongly. [Lesson 30](lessons/30-an-unwanted-behaviour-measured.md)
- **LoRA**: Leaving a grid of weights frozen and training a correction beside it, made of two thin grids multiplied together. [Lesson 26](lessons/26-lora.md)
- **loss**: The model's average surprise over a text. Lower is better. [Lesson 1](lessons/01-predict-the-next-token.md)
- **matrix multiplication**: Written `@`. Output j is each input times the weight in column j, added up. [Lesson 4](lessons/04-a-neural-language-model.md)
- **maximum-likelihood estimate**: The counted table, which gives the training text a higher probability than any other table can. [Lesson 1](lessons/01-predict-the-next-token.md)
- **merge** (of LoRA): Adding a trained correction into its grid, which leaves a plain model. [Lesson 26](lessons/26-lora.md)
- **metrics**: Running totals a server gives as plain text: requests, refusals, tokens and waits. [Lesson 36](lessons/36-the-server.md)
- **MinHash**: Scramble each five-word run into a number and keep the smallest, so two documents agree with chance equal to their overlap. Skips most comparisons. [Lesson 18](lessons/18-data.md)
- **miss**: A story written for an order the rule says to refuse. [Lesson 30](lessons/30-an-unwanted-behaviour-measured.md)
- **momentum**: Keep a velocity for each parameter, multiply it by 0.9, add the new slope, and move along the velocity. [Lesson 11](lessons/11-optimizers.md)
- **move**: One side's turn in a game of chess, written as text such as `Nf3`. [Lesson 38](lessons/38-the-same-code-a-different-world.md)
- **multi-layer perceptron** (MLP): A linear layer, a bend and a second linear layer. Lesson 4's model is one with embeddings in front and softmax behind, and each block of lesson 9 has one inside. [Lesson 4](lessons/04-a-neural-language-model.md)
- **multiplies and adds** (flops): The count of single-number multiplies and adds. Each weight costs 2 for each token in a forward pass. [Lesson 17](lessons/17-what-it-costs.md)
- **nan**: "Not a number", which is what PyTorch gives for inf / inf. [Lesson 3](lessons/03-tensors.md)
- **nat**: The unit of surprise measured with ln. [Lesson 1](lessons/01-predict-the-next-token.md)
- **neural networks**: Models built from layers like the multi-layer perceptron's. [Lesson 4](lessons/04-a-neural-language-model.md)
- **one-hot row**: A token written as a row of numbers that are all 0 except a 1 at its own place. [Lesson 4](lessons/04-a-neural-language-model.md)
- **optimizer**: The rule that turns slopes into steps. [Lesson 11](lessons/11-optimizers.md)
- **order**: Three lines that ask for a story with a given name and three given words. [Lesson 25](lessons/25-fine-tuning.md)
- **overfit**: What a model with a large gap does: it stores its text. [Lesson 13](lessons/13-is-it-working.md)
- **overflow**: A number too big for the fixed count of bits a computer stores it in, such as e to the power 1,000. [Lesson 3](lessons/03-tensors.md)
- **overlap**: The share of their five-word runs that two documents both have: 1 for the same text, near 0 for different stories. [Lesson 18](lessons/18-data.md)
- **overreach**: A refusal of an order the rule says to answer. [Lesson 30](lessons/30-an-unwanted-behaviour-measured.md)
- **padding**: Filling the end of a short row so that rows of different lengths make a grid. Padded places are not scored. [Lesson 25](lessons/25-fine-tuning.md)
- **pair**: Two stories written to one order, with which of them a reader preferred. [Lesson 28](lessons/28-a-model-of-what-people-prefer.md)
- **parameters**: The adjustable numbers in a model's calculation. [Lesson 2](lessons/02-gradients.md)
- **pass at k**: The chance that at least one of k tries passes. [Lesson 24](lessons/24-evaluation.md)
- **pass rate**: A task's mean of passes and failures. [Lesson 24](lessons/24-evaluation.md)
- **percentile**: The 95th percentile of a set of waits is the wait that 1 reader in 20 went over. [Lesson 31](lessons/31-latency-and-throughput.md)
- **piece**: The keys and values of a run of 16 tokens in the store of beginnings, filed under the whole text up to its end. [Lesson 33](lessons/33-prefix-caching.md)
- **place** (position): Where a token stands in the text, counted from the start. [Lesson 8](lessons/08-position.md)
- **prefix caching**: Keeping the keys and values of a beginning that many prompts share, so that each later request reads only what is new. [Lesson 33](lessons/33-prefix-caching.md)
- **probe**: One linear layer trained to read something, here the board, out of the stream of a frozen model. [Lesson 40](lessons/40-the-board-in-its-head.md)
- **quantization**: Storing each weight as a small whole number, with one scale for its row. [Lesson 34](lessons/34-quantization.md)
- **query**: What a token is looking for. [Lesson 6](lessons/06-attention.md)
- **queue**: A line that one thread puts things into and another takes them out of. [Lesson 36](lessons/36-the-server.md)
- **radian**: A unit of angle, in which a full turn is 2π, about 6.283. [Lesson 15](lessons/15-rotary-positions.md)
- **random number generator**: Makes its numbers by a fixed rule from a state it keeps, so the state sets the batches to come. [Lesson 12](lessons/12-the-training-loop.md)
- **rank**: The fewest numbers a grid's input can be passed through. A LoRA correction of rank r passes through r. [Lesson 26](lessons/26-lora.md)
- **rating**: One number for a player's strength, fitted so that the Elo rule makes the results most likely. [Lesson 39](lessons/39-can-it-play.md)
- **readers**: A short function that stands in for people who compare stories. [Lesson 28](lessons/28-a-model-of-what-people-prefer.md)
- **reference**: A frozen copy of the model from before training, against which each story's gain is measured. [Lesson 29](lessons/29-preferences-without-a-scorer.md)
- **refusal**: One fixed line the model gives in place of a story the rule forbids. [Lesson 30](lessons/30-an-unwanted-behaviour-measured.md)
- **residual stream**: The line of plain additions that runs from the first block to the last. [Lesson 9](lessons/09-the-block.md)
- **RMSNorm**: Divide the vector by the root of the mean of its squares, then apply a gain. No mean subtracted and no shift. [Lesson 16](lessons/16-todays-parts.md)
- **rotary positions**: Put the place into each query and key at each attention layer by turning pairs of entries by an angle of place × speed. [Lesson 15](lessons/15-rotary-positions.md)
- **running average**: An average that moves a fixed share of the way toward each new value. [Lesson 11](lessons/11-optimizers.md)
- **safetensors**: A weights file format: 8 bytes holding a length, then that many bytes of JSON naming each tensor, then the raw numbers. [Lesson 10](lessons/10-the-gpt.md)
- **scale** (in quantization): The 32-bit number, one for each row, that says how large a step of 1 is. [Lesson 34](lessons/34-quantization.md)
- **schedule**: The plan for the rate over a run. [Lesson 12](lessons/12-the-training-loop.md)
- **score**: A parameter or output that can be any number, which softmax turns into chances. [Lesson 2](lessons/02-gradients.md) Of a story, the share of the name and the three asked words that it has. [Lesson 27](lessons/27-learning-from-a-score.md)
- **scorer**: A second model that reads an order and its story and gives one number, learned from pairs. [Lesson 28](lessons/28-a-model-of-what-people-prefer.md)
- **seed**: The number that starts the random number generator, fixing the starting weights and the batches. [Lesson 11](lessons/11-optimizers.md)
- **self-attention**: Attention where the queries, keys and values all come from the same tokens. [Lesson 7](lessons/07-causal-multi-head-self-attention.md)
- **server**: A program that waits for requests from other programs and answers each one. [Lesson 36](lessons/36-the-server.md)
- **shape**: How many numbers a tensor has along each direction. [Lesson 3](lessons/03-tensors.md)
- **sigmoid**: 1 / (1 + e to the -x), which squashes any number into the range 0 to 1. [Lesson 16](lessons/16-todays-parts.md)
- **SiLU**: The bend x times sigmoid(x). [Lesson 16](lessons/16-todays-parts.md)
- **slip**: A turn where none of the model's five tries was legal and the referee moved for it. [Lesson 39](lessons/39-can-it-play.md)
- **slope**: The rate at which the loss moves when one number it is made from rises a little. [Lesson 2](lessons/02-gradients.md)
- **softmax**: Raise e to the power of each score, then divide by the row's total, giving positive results that sum to 1. [Lesson 2](lessons/02-gradients.md)
- **speculative decoding**: A small model guesses several tokens, and the big model checks them all in one pass and keeps its own chances exactly. [Lesson 35](lessons/35-speculative-decoding.md)
- **spread** (standard deviation): The usual distance of a set of numbers from their average. [Lesson 6](lessons/06-attention.md)
- **standard error**: The spread of a set of measurements divided by the square root of how many there are. Lesson 14 calls it the error. [Lesson 14](lessons/14-running-an-experiment.md)
- **stochastic gradient descent**: Learning from random batches in place of the whole text. [Lesson 3](lessons/03-tensors.md)
- **stream** (from a server): An answer sent as a run of small events, one for each piece of text. [Lesson 36](lessons/36-the-server.md)
- **stride**: How far the window moves on each time, scoring only the tokens not scored before. [Lesson 24](lessons/24-evaluation.md)
- **stuck**: A hidden unit whose output is beyond 0.99 either way, where tanh's slope is tiny. [Lesson 4](lessons/04-a-neural-language-model.md)
- **surprise**: If the model gave what came next a probability of p, its surprise is -ln p. [Lesson 1](lessons/01-predict-the-next-token.md)
- **sweep**: Trying one setting at a spread of values on a few seeds, such as a rate by factors of about three. [Lesson 14](lessons/14-running-an-experiment.md)
- **SwiGLU**: Widen twice with two grids, pass one result through SiLU, multiply it entry by entry with the other, then narrow with one grid. [Lesson 16](lessons/16-todays-parts.md)
- **tanh**: The function (e^x - e^-x) / (e^x + e^-x), which squashes any number into the range -1 to 1. [Lesson 4](lessons/04-a-neural-language-model.md)
- **task**: A check of one job with a rule that says pass or fail. [Lesson 24](lessons/24-evaluation.md)
- **temperature**: Divide each score by T before softmax. Under 1 likely tokens become likelier, over 1 chances move toward even, and 0 is greedy. [Lesson 22](lessons/22-sampling.md)
- **tensor**: A grid of numbers whose shape says how many lie along each direction. [Lesson 3](lessons/03-tensors.md)
- **thread**: A line of work that runs beside others inside one program. [Lesson 36](lessons/36-the-server.md)
- **throughput**: The tokens written each second, added up over all readers. [Lesson 31](lessons/31-latency-and-throughput.md)
- **time to first token**: The wait before anything appears. [Lesson 31](lessons/31-latency-and-throughput.md)
- **token**: One of the pieces of text a model reads and gives chances to: a character at first, a piece found by byte pair encoding from lesson 5 on. [Lesson 1](lessons/01-predict-the-next-token.md)
- **top-p**: Sort tokens by chance, keep the smallest set whose chances reach p, drop the rest and share their chance among the kept ones. [Lesson 22](lessons/22-sampling.md)
- **training**: Fitting a model to text. [Lesson 1](lessons/01-predict-the-next-token.md)
- **training text**: The part of the text a model learns from. [Lesson 1](lessons/01-predict-the-next-token.md)
- **unseen text**: The part held back from training, on which the loss is reported. [Lesson 1](lessons/01-predict-the-next-token.md)
- **value**: What a token hands over when chosen. [Lesson 6](lessons/06-attention.md)
- **vector**: A row of numbers. [Lesson 6](lessons/06-attention.md)
- **vocabulary**: The set of different tokens. [Lesson 1](lessons/01-predict-the-next-token.md)
- **warmup**: Beginning with a small rate and raising it in a straight line over the first steps. [Lesson 12](lessons/12-the-training-loop.md)
- **weight decay**: On each step each weight grid is multiplied by 1 - rate × decay, so weights drift back toward 0. [Lesson 11](lessons/11-optimizers.md)
- **weight tying**: Using one grid both as the token embeddings and as the last layer that scores tokens. [Lesson 10](lessons/10-the-gpt.md)
- **width**: The count of numbers in a vector. [Lesson 6](lessons/06-attention.md)
- **worker**: A separate process with its own copy of the model and optimizer that finds the slopes for its share of a batch. [Lesson 20](lessons/20-many-workers.md)
