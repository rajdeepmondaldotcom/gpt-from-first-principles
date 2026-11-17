# 18. Data

By the end of this page 210 MB of raw text is 52 million tokens on disk, and no document is in both the training text and the unseen text.

Shakespeare is one megabyte. Lesson 13 showed what a model does with too little text: it stores it. A bigger model needs more text, and we have to download it.

## The idea

Our text is TinyStories: short stories in simple words, written by a large language model for the vocabulary of a small child. The whole set is 2.2 GB. We take the first 209,715,200 bytes (200 × 2²⁰), about 256,000 stories, each followed by lesson 5's end token.

The model trains on junk and repeats like the rest of the text. We take four steps before training.

1. Clean each document. One character can be stored two ways: `é` as one character, or as `e` followed by a separate accent. We rewrite each document so that each character has one form. Then we drop junk: a fragment under 100 characters, a document with broken bytes in it, and a document that is less than 80% letters and spaces.
2. Drop repeats. A document that occurs twice counts twice in training. It can also land once in the training text and once in the unseen text, and then the unseen text is no longer unseen. To find repeats without comparing texts, give each document a fingerprint: a short number worked out from its text, the same for the same text. We take the fingerprint of the text in lowercase with single spaces, and keep the first document with each fingerprint.
3. Drop near-repeats. Change one word and the fingerprint changes. So cut each document into runs of five words in a row. The overlap of two documents is the share of their runs that both have: 1 for the same text, near 0 for different stories. We drop a document whose overlap with an earlier one is 0.8 or more. Comparing each pair of 256,000 documents takes 33 billion comparisons, and MinHash skips most of them. Scramble each run into a number and keep the smallest. Two documents get the same smallest number with a chance equal to their overlap, and a test measures that. We scramble 32 ways and group each document's 32 numbers into 8 bands of 4. Only two documents that agree on a whole band get a real comparison.
4. Split by fingerprint. A document goes to the unseen text when its fingerprint leaves 0 after division by 100. The document's own text decides, so the same text can never be on both sides, and adding documents later moves none of them.

Then the text becomes tokens. We train lesson 5's tokenizer on 5 MB of the training text, up to 4,096 tokens, encode each story with the end token after it, and write the ids to a file, 4 bytes each. Training maps the file into memory, so the machine loads only the parts a batch touches.

## By hand

Two documents of seven words:

- `the cat sat on the mat today`
- `the cat sat on the mat tonight`

Each has three runs of five words. `the cat sat on the` and `cat sat on the mat` are in both. The third run differs. Together they have four different runs and share two, so the overlap is 2/4 = 0.5.

Now the bands. Two documents with an overlap of 0.8 agree on one band of 4 numbers with a chance of 0.8⁴ = 0.41. They miss all 8 bands with a chance of 0.59⁸ = 0.015. So they are compared 98.5% of the time. At an overlap of 0.2 the same sums give 1.3%.

## The code

```python
# gpt/data.py
def clean(doc):
    """One document in, tidy text out. Returns None for junk."""
    doc = unicodedata.normalize("NFC", doc).strip()  # one way to write each character
    plain = sum(ch.isalpha() or ch.isspace() for ch in doc)
    if len(doc) < 100 or "\ufffd" in doc or plain < 0.8 * len(doc):
        return None  # a fragment, broken bytes, or mostly not words
    return doc
```

`\ufffd` is the character Python puts where bytes could not be read as text. Python counts `True` as 1, so `sum` gives the number of letters and spaces.

```python
# gpt/data.py
def minhash(items, a, b):
    """Scramble the items in len(a) different ways. Keep the smallest result of each way."""
    x = torch.tensor(sorted(items)) % PRIME
    return ((a[:, None] * x + b[:, None]) % PRIME).min(dim=1).values
```

`items` are a document's runs, each already turned into a number, and `sorted` makes the set a list that `torch.tensor` can read. One scramble multiplies by a random number, adds another and takes the remainder after division by `PRIME`, which is 2³¹ − 1. `a` and `b` hold 32 such pairs, and broadcasting applies them all at once.

```python
# gpt/data.py
def held_out(doc, one_in=100):
    """Whether a document belongs to the unseen text. Its own fingerprint decides, so it never moves."""
    return int.from_bytes(digest(doc)[:4], "big") % one_in == 0
```

`digest` is the fingerprint, 20 bytes from a standard function called SHA-1. `int.from_bytes` reads its first 4 bytes as one whole number, which is enough to split on.

## The proof

```bash
uv run pytest tests/test_data.py
```

- The test plants a mess: into 200 different speeches from Shakespeare go three repeats (one in capitals, one with other spacing, one exact), two near-repeats (a word changed, a few words added) and three pieces of junk. The pipeline returns the 200 speeches and nothing else.
- `é` written two ways becomes one text.
- The hand example has an overlap of 0.5.
- Two sets with an overlap of 1/3 get the same smallest number on a third of 2,000 scrambles.
- A document is in one split, and shuffling the documents does not move it. Of 600 made-up documents the same four are unseen on any machine: the test holds the list.
- With the bar raised from 0.8 to 1.0, a document with one word added is kept.
- Batches read from the file on disk equal batches from memory.

```bash
uv run python -m gpt.data
```

```
documents read: 256,018
junk: 14, repeats: 0
near-repeats: 0
documents kept: 256,004
train: 253,479 documents, 203.7 MB, 51,919,415 tokens, 3.92 bytes per token
val: 2,525 documents, 2.0 MB, 518,721 tokens, 3.93 bytes per token
longest tokens: [' competitive', ' adventurous', ' disappointed', ' enthusiastic', ' compassionate', ' uncomfortable']
```

The slice was already clean: 14 pieces of junk, no repeats and no near-repeats in 256,018 documents. Text gathered from the web is messier: FineWeb, below, reports what each of its filters removed.

## What it still cannot do

We have 52 million tokens and a model whose size we can set. We do not know which size to train. A small model can take many steps in the time we have. A large one can take few.

## Check yourself

Predict before you run: two documents have an overlap of 0.5. How often do they agree on at least one of the 8 bands?

<details><summary>Answer</summary>

0.5⁴ = 0.0625 for one band. They miss all eight with a chance of 0.9375⁸ = 0.60, so they are compared 40% of the time. We drop a document at an overlap of 0.8 or more, so a pair at 0.5 need not be compared.
</details>

<details><summary>Why split by fingerprint and not by picking each hundredth document?</summary>

Picking by position depends on the order of the file. The same text at two positions could land on both sides, and adding documents would move the rest.
</details>

<details><summary>From lesson 13: a repeat on both sides of the split flatters one number. Which?</summary>

The loss on unseen text. The model has seen that document, so the gap between the two losses looks smaller than it is.
</details>

Go deeper:

- Eldan and Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (2023). The text, and what small models learn from it.
- Hugging Face, [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). The same steps at the scale of the web, with each filter's effect measured.
- Broder, [On the resemblance and containment of documents](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf) (1997). Where MinHash comes from.
