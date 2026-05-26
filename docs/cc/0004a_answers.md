## 1. Why do dot products get larger in higher dimensions?

A dot product is a sum:

```text
q · k = q₁k₁ + q₂k₂ + q₃k₃ + ... + q_dk_d
```

The more dimensions you have, the more terms you add.

The mean stays near zero, but the *spread* (standard deviation) grows as √d_k, meaning individual dot products can be much larger in magnitude than in low dimensions. If the query and key components have roughly variance `1`, then:

```text
Var(q · k) ≈ d_k
```

So as `d_k` grows, the raw attention scores naturally become more spread out.

The standard deviation grows like:

```text
sqrt(d_k)
```

So a 1024-dimensional dot product can easily produce scores roughly tens of units wide, not just tiny values around zero.

---

## 2. What does softmax do to attention scores?

Softmax turns raw scores into a probability distribution that adds up to `1`.

Example:

```text
scores = [1.0, 2.0, 3.0]
```

Softmax might turn that into something like:

```text
weights = [0.090, 0.245, 0.665]
```

In attention, this means:

```text
Token 1 gets 0.090 weighting
Token 2 gets 0.245 weighting
Token 3 gets 0.665 weighting
```

So softmax converts “how strongly does this key match my query?” into “how much of each value should I read?”

It is the step that turns similarity scores into a weighted mixture.

---

## 3. What is softmax collapse?

Softmax collapse happens when one score is so much larger than the others that softmax becomes almost one-hot.

Example:

```text
scores = [1.0, 2.0, 40.0]
```

Softmax becomes basically:

```text
weights = [0.0, 0.0, 1.0]
```

That means one token receives almost all the attention, and the others are ignored.

Healthy attention distributions have one or a few tokens with meaningfully higher weights while others remain nonzero. Collapse means a near-binary distribution where the model has effectively hard-selected one token.

---

## 4. Why does collapse hurt learning?

Because learning depends on smooth, adjustable gradients.

When softmax outputs something like:

```text
[0.0, 0.0, 1.0]
```

there is almost no room to adjust.

The winner is already at `1.0`, so increasing it further is impossible.

The losers are already at `0.0`, so their gradient signal is almost dead.

The model cannot easily learn:

```text
“Actually, token 2 should matter a little more.”
```

or:

```text
“Token 3 should matter slightly less.”
```

So collapse makes attention brittle. It turns a flexible weighting system into a hard switch before the model has learned enough.

---

## 5. Why do we divide by `sqrt(d_k)`?

Because dot product scale grows with dimension.

If:

```text
Var(q · k) ≈ d_k
```

then the typical size of the dot product grows like:

```text
sqrt(d_k)
```

So dividing by `sqrt(d_k)` normalizes the score scale.

```text
scaled_score = (q · k) / sqrt(d_k)
```

Example:

```text
d_k = 1024
sqrt(d_k) = 32
```

A raw score of:

```text
40.0
```

becomes:

```text
40.0 / 32 = 1.25
```

That is much healthier for softmax.

The goal is not to weaken attention. The goal is to keep the numbers in a range where softmax can still make nuanced choices.

---

## 6. What does the Jacobian reveal?

The Jacobian tells us how sensitive the softmax output is to changes in the input scores.

In plain terms, it answers:

```text
“If I slightly change this attention score, how much do the attention weights change?”
```

When softmax is healthy, the Jacobian has meaningful nonzero values. In the non-collapsed case, the diagonal entries are approximately `s_i(1 - s_i)` and off-diagonals are `-s_i s_j`; both are nonzero when the distribution is spread out. That means small score changes can still change the attention distribution.

When softmax collapses, the Jacobian becomes almost all zeros.

That reveals:

```text
small changes to scores no longer matter
```

So the Jacobian proves the training problem mathematically:

> collapsed softmax means dead or nearly dead gradient flow.

---

## 7. In plain English, what does scaling preserve?

Scaling preserves **flexibility**.

It keeps attention from becoming too confident too early.

It preserves:

```text
multiple tokens staying relevant
smooth learning signals
nonzero gradients
the ability to revise attention
```

Without scaling, attention can become:

```text
“Only this token matters.”
```

With scaling, attention becomes:

```text
“This token matters most, but the others can still contribute.”
```

That is the whole point.

**Specifically, it keeps the softmax Jacobian entries at O(s_i(1-s_i)) ≈ O(0.1–0.25) for typical attention weights, rather than collapsing them to O(10⁻¹⁰) or smaller.**

