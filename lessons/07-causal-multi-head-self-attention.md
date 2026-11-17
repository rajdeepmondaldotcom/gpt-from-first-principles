# 7. Causal multi-head self-attention

By the end of this page you have an attention layer that can be trained on each place in a text at once without seeing the answers, and that reads the text in several ways at the same time.

Lesson 6 left two problems. A token can read the token after it, which is the answer it is meant to guess. And each token gets one view of the text.

## The idea

Hide the future. Token i may read tokens 0 to i and nothing later. Before softmax, we set the score for each later token to minus infinity. e to the power of minus infinity is 0, so those tokens get a weight of 0, and the weights that remain still sum to 1.

The rule is called a causal mask: each token's result depends only on what came before it. The layer is called self-attention because the queries, keys and values all come from the same tokens.

The mask buys more than honesty. Lesson 4's model made one guess per window. A masked layer gives a result at each place, and each result can be turned into a guess for the token that follows: after token 0 guess token 1, after tokens 0 and 1 guess token 2, and so on. A window of 8 tokens gives 8 guesses in one pass, and none of them saw its answer.

Now the second problem. Cut each query, key and value into equal slices and run attention on each slice by itself. Each slice is called a head. The numbers in a head's slice come from its own columns of the layer that makes queries, keys and values, so each head has weights of its own and makes its own choice of what to read. Lay the heads' results end to end, and pass them through one more linear layer that mixes them.

At a width of 16, four heads have a width of 4 each. The parameter count does not depend on the number of heads, because the same vectors are cut into more or fewer slices.

## By hand

Take `cat`, the second of three tokens. Say its scores for `the`, `cat` and `sat` are (2, 1, 3). It may not read `sat`.

| | `the` | `cat` | `sat` |
|---|---|---|---|
| Score | 2 | 1 | 3 |
| Masked | 2 | 1 | minus infinity |
| e to that power | 7.389 | 2.718 | 0 |
| Weight (divide by 10.107) | 0.731 | 0.269 | 0 |

The obvious way is to run softmax on (2, 1, 3), set the last weight to 0, and divide the others by their sum. It gives the same weights, and a test checks it. Masking the score does it in one step.

For three tokens the whole mask is a grid. Row i says what token i may read:

```
1 0 0
1 1 0
1 1 1
```

Heads by hand: with two heads, the vector (1, 2, 3, 4) is cut into (1, 2) for the first head and (3, 4) for the second.

## The code

```python
# gpt/attention.py
class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)  # query, key and value for each token, in one layer
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        """x has shape (batch, tokens, width). So does the result."""
        B, T, C = x.shape
        q, k, v = (
            part.view(B, T, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, T, C/heads)
            for part in self.qkv(x).chunk(3, dim=-1)
        )
        past = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()  # row i: tokens 0 to i
        out, _ = attend(q, k, v, past)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))
```

`B`, `T` and `C` are the three directions of `x`: B texts in the batch, T tokens in each, C numbers per token. `self.qkv(x)` makes 3 × C numbers per token, and `chunk(3, dim=-1)` cuts them into the query, the key and the value.

`view(B, T, self.heads, C // self.heads)` cuts each vector into heads. `transpose(1, 2)` then moves the heads next to the batch, giving the shape (batch, heads, tokens, head width). `attend` works on the last two directions, so it runs once and handles each head of each text.

`tril()` keeps the lower-left triangle of a grid of ones, which is the mask above. `device=x.device` builds the mask wherever `x` is stored, and lesson 10 says more about that. `attend` returns the result and the weights, and `_` throws the weights away. The last line lays the heads end to end again and mixes them.

## The proof

```bash
uv run pytest tests/test_attention.py
```

- The masked example gives 0.731, 0.269 and 0, and equals the obvious way.
- A token cannot read the future. Replace token 4 of 6 and the results for tokens 0 to 3 do not change by a single bit.
- Future tokens get a weight of exactly 0, and the first token puts a weight of 1 on itself.
- Heads cut the vector, and each runs `attend` on its slice: the layer equals four separate calls, laid end to end and mixed.
- More heads cost no extra parameters: 16,640 at a width of 64, for 1 head or 8.

## What it still cannot do

In one layer, the last token reads the earlier tokens as a bag. It knows which tokens came before, and not which came first. To the last token of `dog bites man today`, the earlier tokens are the same bag as in `man bites dog today`.

## Check yourself

Predict before you run: a window holds 8 tokens. Lesson 4's model learned from 1 guess per window. How many does this layer give?

<details><summary>Answer</summary>

8, one after each token.
</details>

<details><summary>What are the first token's weights, whatever its scores?</summary>

A weight of 1 on itself. Each other score was set to minus infinity.
</details>

<details><summary>The width is 768 and there are 12 heads. How wide is a head, and what are its scores divided by?</summary>

768 / 12 = 64 wide. The scores are divided by the square root of 64, which is 8.
</details>

<details><summary>Why does replacing token 4 leave tokens 0 to 3 unchanged to the last bit, and not merely close?</summary>

Their weight on token 4 is exactly 0, and 0 times any value is 0. Token 4 takes no part in their sums.
</details>

<details><summary>From lesson 6: the scores are divided by the root of the head's width, not the layer's. Why?</summary>

The dot product runs over one head's slice, so that slice's width sets how far the scores spread.
</details>

Go deeper:

- Vaswani and others, [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.2.2. Multi-head attention as first described.
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY). The mask and the heads, built live.
- Michel, Levy and Neubig, [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650). Many trained heads can be removed with little loss.
