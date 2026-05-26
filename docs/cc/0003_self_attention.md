## CC2 Practice Answers

1. **Two synthetic classes:** Math tasks and Writing tasks.

2. **What `w` represents geometrically:** The direction/normal vector of the separating boundary between the two classes.

3. **Why dot product is used:** It measures how strongly an input vector aligns with the learned class-separating direction.

4. **What bias `b` does:** It shifts the separating boundary without changing its angle.

5. **Why validation data is needed:** To check whether the probe learned the real pattern instead of memorizing the training set.

6. **Why high-dimensional data can be dangerously easy to separate:** In D dimensions, any set of N < D points is almost certainly linearly separable regardless of label assignment, because the constraint count (N) is far less than the degrees of freedom (D). This is a consequence of the geometry of hyperplanes in high dimensions, not merely available 'room.'

7. **What L2 regularization discourages:** Overly large weights and overly complex/fragile separating boundaries.

The script uses Math vs. Writing synthetic clusters, a linear probe with `w`, `b`, sigmoid prediction, L2 weight decay, and unseen validation data. 

---

# Crash Course 3: Self-Attention — How Tokens Talk to Each Other

This maps mainly to `06_self_attention.exs` and `07_softmax_collapse.exs`.

## Core idea

Self-attention lets each token decide (via trained weights that encode this decision function at inference, rather than a per-forward-pass learning event):

> “Which other tokens should I listen to, and how much?”

For a tiny sentence:

```text
"The cat sat"
```

each token starts with its own vector. Self-attention updates each token by mixing in information from the other tokens.

---

## The three roles: Query, Key, Value

Each token is projected into three versions of itself:

| Role  | Meaning                             |
| ----- | ----------------------------------- |
| Query | What am I looking for?              |
| Key   | What information do I contain?      |
| Value | What information do I contribute, weighted by how much others attend to me? |

A token’s **Query** is compared against every other token’s **Key**.

That comparison uses the dot product. Dot product measures directional alignment; tokens with similar Query and Key *directions* will have high scores, which corresponds to semantic compatibility.

---

## Attention scores

The raw attention score is:

```text
score = Query · Key
```

High score means:

> “This token is relevant to me.”

Low score means:

> “This token is less relevant.”

For all tokens, this creates a score matrix:

```text
        Key: The   cat   sat
Query The    ?     ?     ?
Query cat    ?     ?     ?
Query sat    ?     ?     ?
```

*(Note: Each `?` is a scalar: high means that query token should attend strongly to that key token.)*

Each row answers:

> “For this token, how much should I attend to every token?”

---

## Why divide by `sqrt(D_k)`?

If vectors are high-dimensional, raw dot products can get too large.

When scores have high variance — which happens when D_k is large, because dot product variance scales as D_k — softmax amplifies the gap between the max and all others exponentially, collapsing to near-one-hot:

```text
[0.000, 0.999, 0.001]
```

That means the model listens almost entirely to one token. Worse, gradients can vanish.

So attention uses:

```text
scaled_score = (Q · K) / sqrt(D_k)
```

This keeps scores in a healthy range.

---

## Softmax turns scores into attention weights

Softmax converts raw scores into a probability distribution:

```text
[1.2, 4.5, 1.0]
```

becomes something like:

```text
[0.15, 0.70, 0.15]
```

Now the token knows how to distribute its attention.

---

## Values: the actual information being mixed

After attention weights are computed, they are applied to the Value vectors.

```text
output = attention_weights · V
```

So the final output for a token is a weighted mixture of information from the sequence.

---

## One-sentence summary

**Self-attention lets each token use dot products to decide which other tokens matter, softmax to turn those scores into weights, and Value vectors to mix useful information into a new representation.**

Run next:

```bash
elixir 06_self_attention.exs
elixir 07_softmax_collapse.exs
```

