# 10. The GPT

By the end of this page the parts are one model, and on our test prompt that model, running OpenAI's GPT-2 weights, gives GPT-2's answers.

We have embeddings, positions, attention and the block. We have no model, and no evidence that the parts add up to GPT-2.

## The idea

Four steps turn token ids into scores:

1. The embedding gives each token a vector for what it is and where it stands.
2. A stack of blocks works on the vectors.
3. One last layer normalization steadies the result.
4. A linear layer turns each token's vector into one score per token in the vocabulary.

The scores at place i are the model's guess for token i + 1. This model is called a GPT, a generative pre-trained transformer. Generative means it writes. Pre-trained means it is trained once on a great deal of text before being put to other uses. The transformer is the stack of attention blocks.

PyTorch stores the last layer's grid with one row per token in the vocabulary, each as wide as the model. The table of token embeddings has the same shape. GPT-2 uses one grid for both jobs: the vector that says "this token came in" is also the vector a result is compared with to score that token. This is called weight tying. In GPT-2 it saves 38,597,376 parameters out of what would be 163 million.

Each grid starts as random numbers with a spread of 0.02, and each bias at 0. The two layers in a block that add into the residual stream start smaller, divided by the square root of twice the number of blocks. With 12 blocks the stream receives 24 additions, and by lesson 6's rule the spread of a sum of 24 random terms grows like the square root of 24. Dividing by that root keeps the total the same size at any depth.

GPT-2 small has 12 blocks, a width of 768, 12 heads, a context of 1,024 tokens and 50,257 tokens in its vocabulary: 124,439,808 parameters. OpenAI published its trained weights. If our parts are that model, our code with those weights must give the scores that an independent implementation gives.

The weights come in a format called safetensors. A file starts with 8 bytes that hold a length. Then come that many bytes of JSON, a plain-text format, naming each tensor with its shape and where its numbers sit. Then come the raw numbers. We read it by hand. OpenAI stored each linear layer's grid the other way round from PyTorch, so we turn those grids on their side as they load.

Hugging Face's `transformers` library has an implementation of GPT-2. We gave it the prompt `Once upon a time, there was a` and saved what it returned. Our model, with the same weights, picks the same top token at each of the 8 places. At the last place its scores for 31 tokens, the 20 it rates highest and 11 spread across the vocabulary, match the saved ones to within 0.0001, on scores near -100. Its top guess for the next token is ` man`.

## By hand

Count the small model we train below: 65 tokens, a context of 64, a width of 128, 4 blocks.

| Part | Sum | Parameters |
|---|---|---|
| Token rows | 65 × 128 | 8,320 |
| Place rows | 64 × 128 | 8,192 |
| Blocks | 4 × 198,272 | 793,088 |
| Last normalization | 2 × 128 | 256 |
| Scores layer | tied to the token rows | 0 |
| Total | | 809,856 |

One window gives a guess at each place. The window is `the c`, and the targets are the same text one character on, `he ca`. At place 0 the model has read `t` and must guess `h`. At place 4 it has read `the c` and must guess `a`.

## The code

```python
# gpt/gpt2.py
    def __init__(self, vocab=50257, context=1024, width=768, heads=12, layers=12):
        super().__init__()
        self.context = context  # the most tokens the model can read at once
        self.embed = Embedding(vocab, context, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = LayerNorm(width)
        self.out = nn.Linear(width, vocab, bias=False)
        self.out.weight = self.embed.token.weight  # one grid reads tokens and scores them
        # ...
    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))
```

`nn.ModuleList` is a list that PyTorch searches for parameters. `bias=False` leaves the biases out of the last layer, as GPT-2 does. `layers` is the number of blocks, and `# ...` stands for the lines that set the starting weights.

A batch now holds a target for each place, the window moved one token on:

```python
# gpt/gpt2.py
def batch(ids, context, size, generator):
    """Random windows of `context` tokens, and the same windows moved one token on."""
    at = torch.randint(len(ids) - context, (size,), generator=generator).to(ids.device)[:, None]
    steps = torch.arange(context, device=ids.device)
    return ids[at + steps], ids[at + 1 + steps]


def loss_on(model, x, y):
    """One guess per place: the scores at place i are scored against token i + 1."""
    scores = model(x)
    return cross_entropy(scores.view(-1, scores.size(-1)), y.reshape(-1))
```

`view(-1, ...)` lays the places of each window in the batch out as one long list of rows, which is the shape lesson 3's `cross_entropy` takes.

A tensor lives on a device: `cpu`, the processor, or a graphics chip. `.to(ids.device)` puts the positions where the text is.

## The proof

```bash
uv run pytest tests/test_gpt2.py
```

- The parameter count follows the formula: 809,856 for the small model and 124,439,808 for GPT-2.
- One grid reads tokens and scores them.
- The model starts within 0.1 nats of no opinion.
- The whole model cannot read the future: change token 8 and the scores at places 0 to 7 do not change by a bit.
- We can read the weight file format.

For the check against GPT-2, download the weights:

```bash
mkdir -p data/cache
curl -L -o data/cache/gpt2.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
echo "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707  data/cache/gpt2.safetensors" | shasum -a 256 --check
uv run --extra reference pytest -m reference
```

The third line checks that the file is the one we tested. `tests/gpt2_reference.py` is the script that made the saved scores from Hugging Face's code.

Then train the small model with lesson 4's loop, plain gradient descent. We go back to characters: Shakespeare is one megabyte, and 65 tokens keep these runs to minutes. Each step reads 32 windows of 64 characters: 6,144,000 targets in 3,000 steps.

```bash
uv run python -m gpt.gpt2
```

```
parameters: 809,856
loss before training: 4.246 nats (ln 65 = 4.174)
step 500: 2.428 nats
step 1000: 2.394 nats
step 1500: 2.241 nats
step 2000: 2.299 nats
step 2500: 2.154 nats
step 3000: 2.029 nats
unseen text: 2.091 nats, 3.017 bits per byte
```

Each printed loss is for one batch, so it jumps about as it falls.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| Bigram, learned | 3.599 |
| MLP, eight characters back | 2.663 |
| MLP, eight tokens back, 1,024 tokens | 2.572 |
| GPT, 64 characters back | 3.017 |

## What it still cannot do

```
Once upon a timer deat tilses te grin and frit proner bese aff felling Eved.


TORENINA:
You, ha
```

With GPT-2's weights the code gives GPT-2's scores. Trained by us, it has scored three times as many targets as an MLP a twelfth of its size, 6,144,000 against 1,920,000, and still trails it. Lesson 11 keeps the model and changes only the rule that turns slopes into steps.

## Check yourself

Predict before you run: GPT-2 has 124,439,808 parameters stored as 4-byte numbers. How large is the weight file?

<details><summary>Answer</summary>

About 498 MB of parameters. The file is 548 MB because it also stores a copy of the mask for each block, 1,024 × 1,024 numbers each, which are not parameters.
</details>

<details><summary>Without weight tying, how many parameters would the small model have?</summary>

809,856 + 65 × 128 = 818,176. With a vocabulary of 65 tying saves little. With 50,257 it saves 38 million.
</details>

<details><summary>From lesson 7: the loss scores 64 guesses from one window. Why has none of them seen its answer?</summary>

The causal mask. The result at place i is built from tokens 0 to i only, and its answer is token i + 1.
</details>

Go deeper:

- Radford and others, [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019). The GPT-2 paper.
- Andrej Karpathy, [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU). The same model, trained from scratch on rented GPUs.
