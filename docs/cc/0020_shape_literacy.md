# Foundation A: Shape Literacy

This is a missing bridge lesson that should sit **between Crash Course 1/2 and self-attention**.

The goal is simple:

> Learn to read tensor shapes the way a programmer reads function signatures.

Most confusion in ML comes from not knowing what each axis means.

---

# 1. What is a shape?

A tensor’s **shape** tells you how many numbers it contains and how they are organized.

```text
shape: {3}
```

means:

```text
a vector with 3 numbers
```

Example:

```elixir
Nx.tensor([1.0, 2.0, 3.0])
```

Shape:

```text
{3}
```

---

```text
shape: {3, 4}
```

means:

```text
3 rows, 4 columns
```

Example:

```elixir
Nx.tensor([
  [1.0, 0.5, 0.2, 0.1],
  [0.1, 2.0, 0.8, 0.2],
  [0.2, 0.1, 1.5, 1.8]
])
```

Shape:

```text
{3, 4}
```

In the self-attention script, this exact kind of tensor represents:

```text
3 tokens, each with 4 features
```

or:

```text
{seq, dim}
```

The repo’s self-attention lesson uses this structure for `"The cat sat"`: 3 tokens, 4-dimensional embeddings. 

---

# 2. Axis means “which direction?”

For a 2D tensor:

```text
{3, 4}
```

you have two axes:

```text
axis 0 = rows
axis 1 = columns
```

*(Note: some frameworks use different axis conventions; Nx follows the mathematical convention where axis 0 is the outermost/slowest-varying dimension.)*

Example:

```text
[
  [a, b, c, d],
  [e, f, g, h],
  [i, j, k, l]
]
```

Axis 0 moves vertically:

```text
row 1
row 2
row 3
```

Axis 1 moves horizontally:

```text
column 1, column 2, column 3, column 4
```

For ML, axis meanings are usually semantic:

```text
{tokens, features}
{batch, features}
{batch, sequence, features}
{heads, sequence, head_dim}
```

So when you see a shape, ask:

> What does each axis mean?

Not just:

> How many numbers are there?

---

# 3. Common ML shapes

## Single vector

```text
{dim}
```

Example:

```text
{4}
```

Means:

```text
one 4-dimensional vector
```

---

## Sequence of token vectors

```text
{seq, dim}
```

Example:

```text
{3, 4}
```

Means:

```text
3 tokens
4 features per token
```

For:

```text
"The cat sat"
```

you might have:

```text
The → [1.0, 0.5, 0.2, 0.1]
cat → [0.1, 2.0, 0.8, 0.2]
sat → [0.2, 0.1, 1.5, 1.8]
```

Shape:

```text
{3, 4}
```

---

## Batch of examples

```text
{batch, dim}
```

Example:

```text
{500, 2}
```

Means:

```text
500 examples
2 features each
```

That is the kind of shape used in the linear probe lesson, where synthetic examples are 2D vectors and labels are shaped as column targets. 

---

## Batch of sequences

```text
{batch, seq, dim}
```

Example:

```text
{32, 128, 768}
```

Means:

```text
32 examples in the batch
128 tokens per example
768 features per token
```

This is one of the most important Transformer shapes.

---

# 4. Shape literacy rule: name every axis

Never look at this:

```text
{32, 128, 768}
```

and just say:

```text
three-dimensional tensor
```

Say:

```text
{batch=32, seq=128, hidden_dim=768}
```

That one habit removes a huge amount of confusion.

Bad:

```text
x has shape {32, 128, 768}
```

Better:

```text
x has shape {batch=32, seq=128, hidden_dim=768}
```

Best:

```text
x contains 32 sequences, each with 128 tokens, each token represented by a 768-dimensional vector.
```

---

# 5. Matrix multiplication: the shape contract

Matrix multiplication has a contract:

```text
{m, k} × {k, n} → {m, n}
```

The inner dimensions must match.

```text
      {m, k}
          ↓
      must match
          ↑
      {k, n}
```

Example:

```text
{3, 4} × {4, 5} → {3, 5}
```

Why?

Because the shared inner dimension `k` is summed over (contracted); the outer dimensions `m` and `n` survive. The word **contracted** is used to describe this summation — it represents the dimension that disappears.

---

# 6. Dot product as axis contraction

In Nx, `Nx.dot` lets you say which axes to contract.

Contract means:

> Multiply along these axes and sum them away.

Example from self-attention:

```elixir
queries = Nx.dot(x, [1], w_q, [0])
```

Suppose:

```text
x   shape: {3, 4}
w_q shape: {4, 4}
```

The call says:

```text
contract axis 1 of x with axis 0 of w_q
```

So:

```text
x axis 1 has size 4
w_q axis 0 has size 4
```

They match, so the operation is valid.

The contracted `4` disappears, leaving:

```text
x axis 0 → 3
w_q axis 1 → 4
```

Output:

```text
{3, 4}
```

This is equivalent to standard matrix multiplication `x @ w_q` when contracting the inner dimensions; the explicit axis syntax makes the contraction visible.

Meaning:

```text
3 tokens, each projected into 4-dimensional query space
```

---

# 7. The survival rule

When using `Nx.dot`, remember:

> **The Survival Rule:** Contracted axes disappear; uncontracted axes survive.
>
> **Corollary:** The output shape is formed by concatenating the surviving axes in the order they appear: the left tensor's survivors first, then the right tensor's survivors.

Example:

```text
A shape: {3, 4}
B shape: {4, 6}
```

Contract:

```text
A axis 1 with B axis 0
```

Then:

```text
A axis 1 = 4 disappears
B axis 0 = 4 disappears
```

Survivors:

```text
A axis 0 = 3
B axis 1 = 6
```

Output:

```text
{3, 6}
```

That is the whole game.

---

# 8. Shape walkthrough: Q, K, V

Start with:

```text
x shape: {seq=3, dim=4}
```

This means:

```text
3 tokens
4 features per token
```

Projection matrices:

```text
Wq shape: {dim=4, dim=4}
Wk shape: {dim=4, dim=4}
Wv shape: {dim=4, dim=4}
```

Compute:

```text
Q = XWq
K = XWk
V = XWv
```

Shapes:

```text
Q shape: {seq=3, dim=4}
K shape: {seq=3, dim=4}
V shape: {seq=3, dim=4}
```

Same token count. Same feature size. But different learned views of the same tokens.

The script comments describe Q, K, and V as separate projections with resulting tensors shaped `{seq=3, dim=4}`. 

---

# 9. Shape walkthrough: QKᵀ

Attention scores compare every query token against every key token.

```text
Q shape: {seq=3, dim=4}
K shape: {seq=3, dim=4}
```

To compare every query with every key, we need:

```text
QKᵀ
```

Shape:

```text
{3, 4} × {4, 3} → {3, 3}
```

So:

```text
attention_scores shape: {query_seq=3, key_seq=3}
```

That means:

```text
each token gets one score against each token
```

For `"The cat sat"`:

```text
              keys
          The  cat  sat
queries
The        ?    ?    ?
cat        ?    ?    ?
sat        ?    ?    ?
```

Shape:

```text
{3, 3}
```

Entry `scores[i,j]` = dot product of query vector `i` with key vector `j` = alignment score between token `i`'s question and token `j`'s advertisement.

This is why QKᵀ produces a square matrix when the query and key sequence lengths are the same.

---

# 10. Shape walkthrough: attention weights times V

After softmax:

```text
attention_weights shape: {seq=3, seq=3}
```

Values:

```text
V shape: {seq=3, dim=4}
```

Now multiply:

```text
attention_weights × V
```

Shape:

```text
{3, 3} × {3, 4} → {3, 4}
```

The middle `3` contracts.

Output:

```text
{seq=3, dim=4}
```

`Output[i,:]` is a weighted sum of all value vectors, weighted by how much token `i` attended to each token.

Meaning:

```text
each of the 3 tokens receives a new 4-dimensional vector
```

So attention starts with:

```text
{3, 4}
```

and ends with:

```text
{3, 4}
```

Same outer shape, but the token vectors have been context-mixed.

That is a key Transformer pattern:

> preserve the token-vector shape while changing the information inside it.

---

# 11. Shape story of self-attention

Here is the whole flow:

```text
X
{seq=3, dim=4}
```

Project into Q/K/V:

```text
Q = XWq → {3, 4}
K = XWk → {3, 4}
V = XWv → {3, 4}
```

Compare Q to K:

```text
QKᵀ → {3, 3}
```

Softmax:

```text
attention_weights → {3, 3}
```

Mix values:

```text
attention_weights V → {3, 4}
```

Final output:

```text
{seq=3, dim=4}
```

In compact form:

```text
{3,4}
→ Q,K,V each {3,4}
→ QKᵀ {3,3}
→ softmax {3,3}
→ softmax(...)V {3,4}
```

---

# 12. Why `{seq, seq}` matters

The `{seq, seq}` attention matrix is special.

It means:

```text
every token can talk to every token
```

For 3 tokens:

```text
3 × 3 = 9 relationships
```

For 128 tokens:

```text
128 × 128 = 16,384 relationships
```

For 8,192 tokens:

```text
8,192 × 8,192 = 67,108,864 relationships
```

At float32, one attention matrix for one head = 67M × 4 bytes ≈ 268MB. For 32 layers and batch size 1, that's ~8.4GB just in attention intermediates (since 32 × 268MB = 8,576MB ≈ 8.4GB) — which is why FlashAttention recomputes rather than materializes this matrix.

This is why attention can be expensive.

The shape itself tells you the cost.

Shape literacy is also systems literacy.

---

# 13. Linear probe shape walkthrough

The linear probe lesson uses examples shaped like:

```text
x shape: {num_samples, features}
```

For example:

```text
x_train shape: {1000, 2}
```

Meaning:

```text
1000 examples
2 coordinates per example
```

Weights:

```text
w shape: {2, 1}
```

Meaning:

```text
2 input features
1 output logit
```

Bias:

```text
b shape: {1, 1}
```

or scalar-like.

Prediction:

```text
x · w + b
```

Shape:

```text
{1000, 2} × {2, 1} → {1000, 1}
```

Output:

```text
one score per example
```

Labels:

```text
y shape: {1000, 1}
```

So predictions and labels match:

```text
preds shape: {1000, 1}
y shape:     {1000, 1}
```

That is what makes the loss valid.

---

# 14. Broadcasting

Broadcasting means:

> A smaller tensor is automatically stretched across a larger shape.

Example:

```text
logits shape: {1000, 1}
bias shape:   {1, 1}
```

Adding them works because the same bias can be applied to every row.

```text
{1000, 1} + {1, 1} → {1000, 1}
```

Broadcasting is useful, but it can also hide mistakes. For example, if `labels` has shape `{1000}` and `logits` has shape `{1000,1}`, adding them broadcasts to `{1000,1000}` — a silent correctness bug, not an error.

So always ask:

> Did I intend this smaller tensor to be shared across that axis?

---

# 15. SVD shape walkthrough

For a matrix:

```text
W shape: {4, 4}
```

SVD gives:

```text
W = UΣVᵀ
```

Shapes:

```text
U  shape: {4, 4}
Σ  shape: {4, 4}
Vᵀ shape: {4, 4}
```

If we keep only rank 1:

```text
U_r  shape: {4, 1}
Σ_r  shape: {1, 1}
Vᵀ_r shape: {1, 4}
```

Reconstruct:

```text
{4,1} × {1,1} × {1,4}
```

(Intermediate shapes: `{4,1} × {1,1} → {4,1}`; then `{4,1} × {1,4} → {4,4}`)

First:

```text
{4,1} × {1,1} → {4,1}
```

Then:

```text
{4,1} × {1,4} → {4,4}
```

So even a rank-1 reconstruction returns to the original outer shape:

```text
{4,4}
```

But its internal information is constrained to one independent direction.

---

# 16. LoRA shape walkthrough

Original frozen path:

```text
X shape:  {N, D}
W₀ shape: {D, D}
```

Output:

```text
XW₀ᵀ → {N, D}
```

LoRA path:

```text
A shape: {r, D}
Bᵀ shape: {r, D}
```

*(Note: Here `Bᵀ` denotes the transpose of B; B itself has shape `{D,r}` in the Hu et al. convention. The scripts store this as `lora_b` with shape `{r,D}` — i.e., they store `Bᵀ` directly.)*

Using the row-batched convention from the script:

```text
X Aᵀ → {N, r}
```

Then:

```text
{N, r} × {r, D} → {N, D}
```

So LoRA does:

```text
{N, D}
→ compress to {N, r}
→ expand back to {N, D}
```

That is the shape story of LoRA:

> shrink into a small adaptation space, then expand back to the model’s normal hidden dimension.

The LoRA script comments explicitly describe this as `X * A^T` followed by `B^T`, returning a `{N, D}` delta that can be added to the frozen output. 

---

# 17. Shape errors are meaning errors

If your shapes are wrong, it usually means your concept is wrong.

Example:

```text
QKᵀ should be {seq, seq}
```

If you get:

```text
{seq, dim}
```

you probably did not compare all tokens to all tokens.

If LoRA output is:

```text
{N, r}
```

instead of:

```text
{N, D}
```

then you compressed but forgot to expand.

If linear probe output is:

```text
{features, 1}
```

instead of:

```text
{samples, 1}
```

then you probably contracted the wrong axis.

Shapes are not bookkeeping.

Shapes are semantic checks.

---

# 18. The shape-debugging checklist

When confused, ask these in order:

```text
1. What does each axis mean?
2. Which axes are being contracted?
3. Do the contracted axes have the same size?
4. Which axes survive?
5. Does the output shape match the concept?
6. Is broadcasting happening?
7. If broadcasting is happening, did I intend it?
```

That checklist catches most tensor bugs.

---

# 19. Core examples to memorize

## Linear layer

```text
X {batch, input_dim}
W {input_dim, output_dim}
→ Y {batch, output_dim}
```

## Self-attention

```text
X {seq, dim}
Q,K,V {seq, dim}
QKᵀ {seq, seq}
softmax(QKᵀ)V {seq, dim}
```

## Batched Transformer input

```text
X {batch, seq, hidden_dim}
```

## Multi-head attention preview

```text
X {batch, seq, hidden_dim}
→ split into heads
{batch, heads, seq, head_dim}
```

*(Note: This split is a reshape + transpose: `hidden_dim = heads × head_dim`; the hidden dimension is partitioned into heads, then axes are rearranged so each head sees the full sequence.)*

where:

```text
hidden_dim = heads × head_dim
```

## LoRA

```text
X {N, D}
A {r, D}
Bᵀ {r, D}
X Aᵀ {N, r}
(X Aᵀ) Bᵀ {N, D}
```

---

# 20. Practice questions

Answer these before continuing:

1. What does shape `{3, 4}` mean in the self-attention script?
2. What is an axis?
3. In `{batch, seq, dim}`, what does each axis mean?
4. What does it mean to contract an axis?
5. In `{3, 4} × {4, 5} → {3, 5}`, which dimension disappears?
6. Why does `QKᵀ` produce `{seq, seq}`?
7. Why does `attention_weights × V` return `{seq, dim}`?
8. What does broadcasting do?
9. Why is the `{seq,seq}` attention matrix expensive in both memory and compute for long sequences? What grows quadratically?
10. Why are shapes semantic, not just mechanical?

---

# One-sentence summary

**Shape literacy means knowing what every tensor axis represents, which axes are being summed away, and whether the output shape matches the meaning of the operation.**
