# 5. Tokens

By the end of this page the same window of eight reaches more than twice as far back.

Eight characters is about one word. To see further with the same model, make each piece of text bigger.

## The idea

The obvious bigger piece is a word. The training text has 23,841 different words. The unseen text has 20,153 words, and 2,361 of them never occur in training. A model with a row per word has no row for those. It can neither read them nor write them, and more than one word in nine is a hole.

Characters never run out, and they are too small. We want pieces that are as large as words where the text allows and can still spell anything.

Start from bytes. A computer stores text as bytes, numbers from 0 to 255. Plain English letters take one byte each, `é` takes two, and an emoji takes four. Any text in any language is a row of bytes, so a vocabulary of 256 tokens, one per byte, can write anything.

Now grow the vocabulary. Find the pair of neighbouring tokens that occurs most often in the training text. Make that pair a new token, and replace it wherever it occurs. Repeat until the vocabulary is the size you want. This is byte pair encoding, or BPE.

Common words end up as one token. Rare words stay in pieces. Nothing is ever unknown, because the 256 bytes are still in the vocabulary.

To encode new text, start from its bytes and keep joining the pair that was learned earliest, until no learned pair is left.

Cut the text into chunks first: a run of letters, a run of digits or a run of anything else, each with the one space before it. Merges happen inside a chunk, so no token spans two words. The usual token for a word in the middle of a sentence carries its space in front, as in ` time`.

Keep one extra token that marks where a document ends. It never comes out of ordinary text. If a user types the characters `<|endoftext|>`, they are encoded as plain characters, or anyone could forge the end of a document.

Expect half characters. A token can hold one byte of a two-byte character. Decoded alone, that token is not text, and Python prints a placeholder for it.

Bigger tokens are a trade. The same window covers more text. In return the model has more rows to learn, and sees each one less often.

Models with different tokens cannot be compared by loss per token, because a bigger token carries more text. So we convert to bits per byte: the loss per token, divided by ln 2 and by the average bytes in a token. That is why the scoreboard has used bits per byte since lesson 1.

## By hand

Train on `aaabdaaabac`. Each character is one byte, so we start with 11 tokens.

| Tokens | Pair counts | Merge |
|---|---|---|
| a a a b d a a a b a c | aa 4, ab 2, four others 1 | aa becomes Z |
| Z a b d Z a b a c | Za 2, ab 2, four others 1 | a tie: take the smaller pair, ab, as Y |
| Z Y d Z Y a c | ZY 2, four others 1 | ZY becomes X |
| X d X a c | | |

Three merges took 11 tokens down to 5. X stands for `aaab`.

To encode `aab` with this vocabulary, start from a a b. The earliest merge that applies is aa, giving Z b. No learned pair is left, so the result is two tokens. The second merge, ab, never ran, because aa came first and used up the `a`.

## The code

```python
# gpt/bpe.py
    while len(vocab) < size:
        pairs = Counter()
        for tokens, n in chunks.items():
            for pair in pairwise(tokens):
                pairs[pair] += n
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))  # most frequent; ties to the smallest
        vocab[best[0] + best[1]] = len(vocab)
        chunks = {merge(tokens, best): n for tokens, n in chunks.items()}
```

`chunks` maps each different chunk of the text, as a row of tokens, to how often it occurs. Counting each chunk once and multiplying by `n` saves going through the whole text on each merge. `vocab` maps the bytes of a token to its id, so a new token's id is the size of the vocabulary so far.

```python
# gpt/bpe.py
    def encode_chunk(self, chunk):
        """Start from bytes. Keep joining the pair that was learned earliest."""
        parts = [bytes([b]) for b in chunk.encode()]
        while len(parts) > 1:
            rank, i = min(
                (self.vocab.get(a + b, math.inf), i) for i, (a, b) in enumerate(pairwise(parts))
            )
            if rank == math.inf:
                break
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        return [self.vocab[p] for p in parts]
```

A token learned earlier has a smaller id, so the smallest id among the joinable pairs is the earliest merge.

## The proof

```bash
uv run pytest tests/test_bpe.py
```

- The hand example learns aa, ab and aaab, in that order, and encodes the text as 5 tokens.
- The chunks cover any text exactly, with nothing lost between them.
- Any text comes back unchanged from encode then decode: accents, Japanese, emoji, tabs.
- Encoding gives the same tokens as replaying the merges one by one in the order they were learned.
- More merges mean fewer tokens.
- The end token comes only when asked for.
- Half of `é` decodes to a placeholder, and both halves to `é`.
- Given GPT-2's vocabulary, our encoder returns the same ids as OpenAI's tiktoken. This test needs `uv sync --extra reference`.

```bash
uv run python -m gpt.bpe
```

```
words in the training text: 23,841 different ones
words in the unseen text: 20,153, of which 2,361 never occur in training
512 tokens: 1.90 bytes per token, MLP scores 2.597 bits per byte
1024 tokens: 2.28 bytes per token, MLP scores 2.572 bits per byte
first merges: [' t', 'he', ' a', 'ou', ' s', ' m', 'in', ' w', 're', 'ha', ' the', 'nd']
longest tokens: [' against', ' nothing', 'ORIOLANUS', 'GLOUCESTER', ' VINCENTIO', 'CORIOLANUS']
'Once upon a time' -> ['O', 'nce', ' upon', ' a', ' time']
```

The script trains lesson 4's model again, unchanged except that it reads eight tokens in place of eight characters.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| MLP, eight characters back | 2.663 |
| MLP, eight tokens back, 1,024 tokens | 2.572 |

## What it still cannot do

With 1,024 tokens the window reaches 18 bytes, about three words. The score moved from 2.663 to 2.572, which is far less than the jump from one character of context to eight.

Reach is no longer the main limit. Eight tokens are still eight fixed slots, each read by weights of its own. What the model learns about a word in the third slot does not help it when the same word sits in the fifth.

## Check yourself

Predict before you run: training starts at 256 tokens and stops at 1,024. How many merges is that?

<details><summary>Answer</summary>

768. Each merge adds one token.
</details>

<details><summary>With the hand example's vocabulary, how many tokens is `aaaa`?</summary>

Two. a a a a becomes Z a a, then Z Z. Each step joins the leftmost pair that the earliest merge covers.
</details>

<details><summary>A model scores 4.07 nats per token and a token averages 2.28 bytes. What is that in bits per byte?</summary>

4.07 / 0.693 / 2.28 = 2.57.
</details>

<details><summary>Why does the token for a mid-sentence word begin with a space?</summary>

Each chunk carries the space before it. The model then learns one token for ` time` and does not spend a separate token on each space.
</details>

<details><summary>From lesson 4: the vocabulary grew from 65 to 1,025 with the end token. Which parts of the model grew?</summary>

The embedding table, to 1,025 rows, and the second layer, to 1,025 scores. The first layer did not change.
</details>

Go deeper:

- Sennrich, Haddow and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (2016). BPE brought to language models.
- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). The same algorithm, with GPT-4's chunk rules.
- OpenAI, [tiktoken](https://github.com/openai/tiktoken). The encoder our test compares against.
