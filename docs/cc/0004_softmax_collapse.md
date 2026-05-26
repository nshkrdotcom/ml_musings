# Crash Course 4: Softmax Collapse — Why Attention Needs Scaling

This follows naturally from Q/K/V and maps to `07_softmax_collapse.exs`. The script shows why attention uses:

```text
QKᵀ / sqrt(d_k)
```

instead of just:

```text
QKᵀ
```

The big idea:

> Dot products get larger as vector dimensions grow. If attention scores get too large, softmax becomes too confident, and learning breaks.

The lesson scripts demonstrate this by generating high-dimensional attention scores, comparing unscaled vs. scaled softmax, and then computing the softmax Jacobian to show gradient collapse directly. 

---

# 1. The problem: big vectors make big dot products

A dot product adds many little multiplications:

```text
q · k = q₁k₁ + q₂k₂ + q₃k₃ + ... + q_Dk_D
```

If `D` is small, the sum stays modest.

If `D = 1024`, you are adding 1024 terms.

Even if each term is random-ish, the total score can get large.

So attention scores can look like:

```text
[3.2, -8.5, 41.0, 2.1, -17.3]
```

That `41.0` dominates everything.

---

# 2. Softmax turns scores into probabilities

Softmax takes scores and turns them into weights that sum to 1:

```text
[score1, score2, score3]
        ↓
[weight1, weight2, weight3]
```

Example:

```text
[1.0, 2.0, 3.0]
```

might become:

```text
[0.09, 0.24, 0.67]
```

That is healthy. Multiple tokens still matter.

But if the scores are too spread out:

```text
[1.0, 2.0, 40.0]
```

softmax becomes:

```text
[0.0, 0.0, 1.0]
```

Now one token gets everything.

That is **softmax collapse**.

---

# 3. Why collapse is bad

At first, collapse sounds useful:

> “Great, the model found the important token.”

But during training, it is dangerous.

If softmax becomes almost exactly:

```text
[0, 0, 1, 0, 0]
```

then the model stops being able to smoothly adjust attention.

It becomes too certain too early.

The gradient shrinks toward zero.

So the model cannot easily learn:

```text
“Maybe token 2 should get a little more attention.”
```

The attention distribution becomes rigid.

---

# 4. The fix: divide by `sqrt(d_k)`

Transformer attention uses:

```text
scores = QKᵀ / sqrt(d_k)
```

Where `d_k` is the key/query dimension.

If `d_k = 1024`, then:

```text
sqrt(1024) = 32
```

So a raw score like:

```text
41.0
```

becomes:

```text
41.0 / 32 = 1.28
```

Now softmax stays smooth.

Instead of:

```text
[0.0, 0.0, 1.0]
```

you get something more like:

```text
[0.12, 0.18, 0.55, 0.10, 0.05]
```

One token can still be strongest, but the others are not dead.

---

# 5. The core intuition

Scaling does not change the logic of attention.

It changes the **temperature**.

Unscaled attention is like a person who instantly shouts:

```text
“ONLY THIS TOKEN MATTERS.”
```

Scaled attention says:

```text
“This token matters most, but keep the others in play.”
```

That “keep the others in play” part is what preserves learning.

---

# 6. Why `sqrt(d_k)` specifically?

Because the variance of the raw dot product grows with dimension:

```text
Var(q · k) ≈ d_k
```

The standard deviation grows as:

```text
sqrt(d_k)
```

So dividing by `sqrt(d_k)` brings the score scale back to a stable range.

Not too hot.
Not too cold.
Trainable.

---

# 7. What the script teaches

`07_softmax_collapse.exs` does four important things:

1. Creates high-dimensional random attention scores.
2. Applies softmax without scaling.
3. Applies softmax with scaling.
4. Computes the Jacobian to show that the unscaled case has nearly dead gradients.

That last part matters.

It does not merely say:

> “Scaling helps.”

It shows:

> “Without scaling, the derivative of softmax collapses.”

---

# 8. The sticky summary

Think of attention as a voting system.

Raw `QKᵀ` gives each token a score.

Softmax turns scores into votes.

But if the scores are too extreme, one token gets all the votes and everyone else gets silenced.

Scaling by `sqrt(d_k)` keeps the vote competitive.

```text
QKᵀ                 = raw match scores
QKᵀ / sqrt(d_k)     = stabilized match scores
softmax(...)        = attention percentages
softmax(...) V      = blended information
```

---

# Practice questions

Answer these before moving on:

1. Why do dot products get larger in higher dimensions?
2. What does softmax do to attention scores?
3. What is softmax collapse?
4. Why does collapse hurt learning?
5. Why do we divide by `sqrt(d_k)`?
6. What does the Jacobian reveal?
7. In plain English, what does scaling preserve?

Run:

```bash
elixir 07_softmax_collapse.exs
```

