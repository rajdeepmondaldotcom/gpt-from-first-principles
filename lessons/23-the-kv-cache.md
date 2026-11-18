# 23. The KV cache

By the end of this page the model reads each token once while it writes, and the story it writes is the same.

To write each new token, the model reads the whole text again from its first token. A story of 200 tokens is read 200 times, each time one token longer.

## The idea

Add one token to the end. Lesson 7's mask lets a token read only the tokens before it, so the results for the first 200 tokens are the same whether or not a 201st follows. We need not work them out again.

The new token does need something from the old ones. In each block, attention scores the new token's query against the key of each earlier token and mixes their values. It needs nothing else from them: only their own results use their queries and what their MLPs made.

So keep, in each block, the key and the value of each token read so far. This is the KV cache. To add a token, run that one token through the model. In each block it makes its query, key and value, puts the key and value at the end of the cache, and reads the whole cache.

For a text of n tokens the model then reads n tokens in all. Without the cache it reads 1 + 2 + ... + n, about n² / 2.

A cache goes wrong in three places, and each has a test here.

- The new token's place is the number of tokens in the cache. A cache that starts each call at place 0 turns the new query and key by the wrong angle, and the wrong key stays in the cache.
- A prompt can arrive in pieces, so one call can bring several new tokens. The new tokens need lesson 7's mask among themselves: each reads the whole cache, and the new tokens up to itself.
- With grouped heads the cache holds one key for each key head, and lesson 16 repeats them for the query heads at the moment of use.

The cache costs memory, and lesson 17 counted it. The story model has 6 blocks and 4 key heads of 32 numbers, so it keeps 2 × 6 × 4 × 32 × 4 = 6,144 bytes for each token: a key and a value, in 6 blocks, 4 key heads of 32 numbers, 4 bytes each. A story of 250 tokens holds 1.5 MB. At 6,144 bytes a token, the cache matches the 21.5 MB of weights at about 3,500 tokens: about fourteen full contexts at once.

## By hand

The prompt `Once upon a time` is 4 tokens, and we write 200 more.

With the cache the model reads the 4, then each new token once except the last, which nothing follows.

4 + 199 = 203 tokens

Without it the model reads 4 tokens for the first new one, 5 for the second, and 203 for the last.

4 + 5 + ... + 203 = 200 × (4 + 203) / 2 = 20,700 tokens

## The code

Lesson 15 quoted attention with two names left for later. `seen` is how many tokens the cache holds, and it sets the new tokens' places:

```python
# gpt/model.py
        seen = cache["k"].size(2) if cache else 0  # tokens read before this call
        pos = torch.arange(seen, seen + T, device=x.device)
```

The new keys and values join the old ones, and the cache keeps the result:

```python
# gpt/model.py
        if cache is not None:
            if cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
```

`cache` is a dictionary for one block, empty until the first call. The mask has a row for each new token and a column for each token old or new:

```python
# gpt/model.py
        past = torch.ones(T, seen + T, dtype=torch.bool, device=x.device).tril(seen)
```

`tril(seen)` keeps, in row i, the first `seen` + i + 1 columns: the whole cache and the new tokens up to this one.

The loop that writes hands the model only what it has not read. The model's second argument, `kept`, is a list with one cache for each block:

```python
# gpt/generate.py
@torch.no_grad()
def generate(model, ids, n, stop=None, cache=True, **how):
    """Write up to n tokens after `ids`, stopping at the context the model trained on.

    With a cache, the model reads each token once.
    """
    ids, device = list(ids), model.embed.weight.device
    kept = [{} for _ in model.blocks] if cache else None
    unread = ids
    for _ in range(min(n, model.context - len(ids))):
        scores = model(torch.tensor([unread if cache else ids], device=device), kept)[0, -1]
        ids.append(choose(scores, **how))
        if ids[-1] == stop:
            break
        unread = ids[-1:]
    return ids
```

## The proof

```bash
uv run pytest tests/test_model.py tests/test_generate.py
```

- Twelve tokens read in one piece, one at a time, in pieces of 5, 1, 4 and 2, or in pieces of 3 and 9 give the same scores as one pass with no cache.
- The cache then holds 12 tokens for each of 2 key heads.
- With and without a cache, the text written is the same.
- With a cache the model reads the prompt and then each new token but the last. Without one it reads 250 tokens to write 20 after a prompt of 3.

Three planted bugs each turn a test red: a mask that forgets the cache, places that start again at 0, and a cache that keeps only its newest tokens.

```bash
uv run python -m gpt.generate
```

```
without a cache: 20,700 tokens read, 227 tokens a second
with the cache: 203 tokens read, 802 tokens a second
the two stories are the same: True
the cache holds 6,144 bytes for each token
```

Timings are from an M4 laptop, and they move from run to run.

With the cache the model reads 102 times fewer tokens and writes 3.5 times as fast. It still calls the model 200 times, once for each token.

## What it still cannot do

The model now writes fast enough to be judged on hundreds of stories. We cannot yet say how sure a loss is, or whether a second model is better than this one.

## Check yourself

Predict before you run: a prompt of 10 tokens, and 100 tokens to write. How many tokens does the model read with a cache, and without?

<details><summary>Answer</summary>

With: 10 + 99 = 109. Without: 10 + 11 + ... + 109 = 100 × (10 + 109) / 2 = 5,950.
</details>

<details><summary>Why does the cache keep keys and values and not queries?</summary>

A token's query is used once, to work out its own result. Its key and value are read again by each token that comes after.
</details>

<details><summary>How much memory does the story model's cache need for a full context of 256 tokens?</summary>

6,144 bytes a token: 1,572,864 bytes, about 1.6 MB.
</details>

<details><summary>From lesson 7: why does adding a token leave the earlier results as they were?</summary>

The causal mask. The result at a place is built from that token and the ones before it, so a later token cannot reach it.
</details>

Go deeper:

- Chen, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). What the cache saves and what it costs, in numbers.
- Hugging Face, [Caching](https://huggingface.co/docs/transformers/kv_cache). The same cache in a library many models share.
