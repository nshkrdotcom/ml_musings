## 1. What does matrix rank measure?

**Rank measures how many independent directions of information a matrix contains.**

A matrix can have many rows and columns, but if they are just copies or scaled versions of each other, the matrix does not contain much independent structure.

So:

```text
high rank = many independent directions
low rank  = few independent directions
```

---

## 2. Why can a large matrix still be low-rank?

Because size is not the same as information.

Example:

```text
[1  2  3]
[2  4  6]
[3  6  9]
```

This is a 3×3 matrix, but every row is just a scaled version of the first row.

It looks big, but it only has **one real direction**.

So it is rank 1.

---

## 3. What does SVD reveal?

**SVD reveals the hidden directions inside a matrix.**

It decomposes a matrix into:

```text
W = U Σ Vᵀ
```

Meaning:

```text
Vᵀ = input directions
Σ  = strength of each direction
U  = output directions
```

It tells us:

> “Here are the main ways this matrix transforms space.”

---

## 4. What do singular values tell us?

**Singular values tell us how important each direction is.**

Large singular value:

```text
this direction matters a lot
```

Small singular value:

```text
this direction matters little
```

Zero singular value:

```text
this direction carries no independent information
```

So singular values are like an importance ranking for the matrix’s hidden directions.

---

## 5. Why does rank-1 compression work on the redundant matrix?

Because the redundant matrix really only has one independent pattern.

Its rows are scaled copies of each other:

```text
row 2 = 2 × row 1
row 3 = 3 × row 1
row 4 = 4 × row 1
```

So keeping only the strongest singular direction preserves essentially everything.

Rank-1 compression works because the matrix is already rank 1.

---

## 6. Why does rank-1 compression fail on the random matrix?

Because a random matrix usually has many independent directions.

Its rows and columns are not simple copies of each other.

So forcing it into rank 1 throws away real information.

Rank-1 compression says:

```text
keep only one direction
discard the rest
```

That is fine for a redundant matrix, but bad for a full-rank random matrix.

---

## 7. What does LoRA freeze?

LoRA freezes the original pretrained weight matrix:

```text
W₀
```

That means the model’s main learned knowledge is left unchanged.

The base model is not directly retrained.

---

## 8. What does LoRA train?

LoRA trains a small low-rank update path:

```text
ΔW = B A
```

Instead of changing the full original matrix, it learns two smaller matrices:

```text
A = down-projection
B = up-projection
```

Together, they create a small update that is added to the frozen model’s output.

---

## 9. Why is `ΔW = BA` cheaper than training a full `ΔW`?

A full update matrix might be:

```text
D × D
```

That is huge.

LoRA uses:

```text
A has shape {r, D}
B has shape {D, r}
```

where `r` is much smaller than `D`.

So instead of training:

```text
D² parameters
```

LoRA trains roughly:

```text
2Dr parameters
```

Example:

```text
D = 4096
r = 8
```

Full update:

```text
4096 × 4096 = 16,777,216 parameters
```

LoRA update:

```text
A has shape {r=8, D=4096} = 32,768 parameters
B has shape {D=4096, r=8} = 32,768 parameters
Total = 65,536 parameters
```

Much cheaper.

---

## 10. In plain English, what is LoRA doing?

LoRA says:

> “Do not relearn the whole model. Keep the original model frozen, and learn a small side adjustment.”

It adds a small trainable adapter that nudges the model’s behavior in useful directions.

So instead of replacing the model’s knowledge, LoRA lightly steers it.

**Plain English:** LoRA is a cheap, low-rank steering patch for a large frozen model. This works in practice because fine-tuning changes tend to be approximately low-rank — the gradient updates during task-specific training concentrate in a small subspace of the full parameter space.
