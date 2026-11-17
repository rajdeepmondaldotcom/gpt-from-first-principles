# 5. Tokens

By the end of this page the same window of eight reaches more than twice as far back.

Eight characters is about one word. To see further with the same model, make each piece of text bigger.

## The idea

The obvious bigger piece is a word. Call a word a run of letters, and ignore capitals. The training text has 10,817 different words. The unseen text is 20,724 words long, and 1,068 of those never occur in training. A model with a row per word has no row for those. It can neither read them nor write them: one word in nineteen is a hole.

Characters never run out, and they are too small. We want pieces as large as words where the text allows, that can still spell anything.

A computer stores text as bytes, numbers from 0 to 255. Plain English letters take one byte each, `é` takes two, and an emoji takes four. Any text in any language is a row of bytes, so a vocabulary of 256 tokens, one per byte, can write anything.

Now grow the vocabulary. Find the pair of neighbouring tokens that occurs most often in the training text. Make that pair a new token, and replace it wherever it occurs. Repeat until the vocabulary is the size you want. This is byte pair encoding, or BPE.

Common words end up as one token. Rare words stay in pieces. Nothing is ever unknown, because the 256 bytes are still in the vocabulary.

Turning text into token ids is called encoding, and turning ids back into text is decoding. To encode new text, start from its bytes. Find the pair of neighbours that was learned earliest, join its leftmost occurrence, and repeat until no learned pair is left.

Cut the text into chunks first: a run of letters, a run of digits or a run of anything else, each with the one space before it. Merges happen inside a chunk, so no token spans two words. The usual token for a word in the middle of a sentence carries its space in front, as in ` time`.

Keep one extra token that marks where a document ends. It never comes out of ordinary text. If a user types the characters `<|endoftext|>`, they are encoded as plain characters, or anyone could forge the end of a document.

A token can hold one byte of a two-byte character. Decoded alone it is not text, and Python prints a placeholder for it.

Bigger tokens are a trade. The same window covers more text. In return the model has more rows to learn, and sees each one less often.

Models with different tokens cannot be compared by loss per token, because a bigger token carries more text. So we convert to bits per byte: the loss per token, divided by ln 2 and by the average bytes in a token.

## By hand

Train on `aaabdaaabac`. Each character is one byte, so we start with 11 tokens.

| Tokens | Pair counts | Merge |
|---|---|---|
| a a a b d a a a b a c | aa 4, ab 2, four others 1 | aa becomes Z |
| Z a b d Z a b a c | Za 2, ab 2, four others 1 | a tie: take the pair that sorts first, ab, as Y |
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
        # shortcut: rescans the chunk after each join, so a chunk of n bytes costs n squared;
        # fine for words, replace before encoding text from strangers.
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

`vocab.get(a + b, math.inf)` is the id of the token a pair would make, or infinity if no such token was learned. A token learned earlier has a smaller id, so `min` finds the earliest merge, and the leftmost one when there are several. Ties in training go to the pair that sorts first as bytes, first token then second. Capitals sort before lowercase.

## The proof

```bash
uv run pytest tests/test_bpe.py
```

- The hand example learns aa, ab and aaab, in that order, and encodes the text as 5 tokens.
- The chunks cover any text exactly.
- Any text comes back unchanged from encode then decode: accents, Japanese, emoji, tabs.
- Encoding gives the same tokens as replaying the merges one by one in the order they were learned.
- The end token comes only when asked for.
- Half of `é` decodes to a placeholder, and both halves to `é`.
- GPT-2 was trained with BPE. Given its vocabulary, our encoder returns the same ids as tiktoken, OpenAI's own encoder. This test runs with `uv run --extra reference pytest -m reference`.

```bash
uv run python -m gpt.bpe
```

```
words in the training text: 10,817 different ones
words in the unseen text: 20,724, of which 1,068 never occur in training
512 tokens: 1.90 bytes per token, MLP 2.597 bits per byte
1024 tokens: 2.28 bytes per token, MLP 2.572 bits per byte
first merges: [' t', 'he', ' a', 'ou', ' s', ' m', 'in', ' w', 're', 'ha', ' the', 'nd']
longest tokens: [' against', ' nothing', 'ORIOLANUS', 'GLOUCESTER', ' VINCENTIO', 'CORIOLANUS']
'Once upon a time' -> ['O', 'nce', ' upon', ' a', ' time']
```

The script trains lesson 4's model again on eight tokens in place of eight characters.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| Bigram, learned | 3.599 |
| MLP, eight characters back | 2.663 |
| MLP, eight tokens back, 1,024 tokens | 2.572 |

## What it still cannot do

With 1,024 tokens the window reaches 18 bytes, about three words. The loss moved from 2.663 to 2.572 bits per byte, which is far less than the jump from one character of context to eight.

More than doubling the reach bought little. Eight tokens are still eight fixed slots, each read by weights of its own.

## Check yourself

Predict before you run: training starts at 256 tokens and stops at 1,024. How many merges is that?

<details><summary>Answer</summary>

768. Each merge adds one token.
</details>

<details><summary>With the hand example's vocabulary, how many tokens is `aaaa`?</summary>

Two. a a a a becomes Z a a, then Z Z. Each step joins the leftmost pair that the earliest merge covers.
</details>

<details><summary>A model's loss is 4.072 nats per token and a token averages 2.284 bytes. What is that in bits per byte?</summary>

4.072 / 0.6931 / 2.284 = 2.572.
</details>

<details><summary>From lesson 4: the vocabulary grew from 65 to 1,025 with the end token. Which parts of the model grew?</summary>

The embedding table, to 1,025 rows, and the second layer, to 1,025 scores. The first layer did not change.
</details>

Go deeper:

- Sennrich, Haddow and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (2016). BPE brought to neural translation.
- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). The same algorithm, with GPT-4's chunk rules.
- OpenAI, [tiktoken](https://github.com/openai/tiktoken). The encoder our test compares against.
