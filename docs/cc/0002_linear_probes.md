# Crash Course 2: Linear Probes, Separating Lines, and “Meaning as Geometry”

Now that Course 1 covered tensors, compiled math, vectors, dot products, and high-dimensional geometry, the next step is:

> How do we prove that a model’s vector space actually contains usable meaning?

That is what **linear probing** teaches. This maps mainly to `05_linear_probe.exs`, `lesson_2_notes.txt`, and the representation-learning sections of the docs. 

---

# 1. The Core Idea

A neural network does not store meaning as English sentences internally.

It stores meaning as vectors.

So instead of asking:

> “Does the model understand math vs. writing?”

we ask:

> “Are math-like vectors and writing-like vectors arranged differently in space?”

A **linear probe** tests this by training the simplest possible classifier on top of the vectors.

If a simple line can separate the categories, then the information was already organized in the representation space.

---

# 2. The Simple 2D Version

Imagine two clusters of points:

```text
Math tasks:     around [ 2.0,  2.0]
Writing tasks:  around [-2.0, -2.0]
```

Visually:

```text
        y
        ↑
  Math  ● ● ●
        ● ● ●

----------------→ x

        ○ ○ ○
        ○ ○ ○  Writing
```

A linear probe tries to draw one separating boundary:

```text
        y
        ↑
  Math  ● ● ●
        ● ● ●
           \
------------\---→ x
             \
        ○ ○ ○ \
        ○ ○ ○  Writing
```

That line is the classifier.

In higher dimensions, the “line” becomes a **hyperplane**, but the idea is the same.

---

# 3. The Formula

The probe computes:

```text
z = w · x + b
```

Where:

| Symbol  | Meaning                                                 |
| ------- | ------------------------------------------------------- |
| `x`     | input vector / embedding                                |
| `w`     | learned direction that separates the classes            |
| `b`     | bias / offset                                           |
| `w · x` | dot product alignment between input and class direction |
| `z`     | raw score before probability                            |

Then it applies sigmoid:

```text
prediction = sigmoid(z)
```

The sigmoid turns the raw score into a probability between `0.0` and `1.0`.

So the model says:

```text
near 0.0 → Class 0, Math
near 1.0 → Class 1, Writing
```

---

# 4. What the Probe Is Really Learning

The weights `w` are not just “parameters.”

Geometrically, `w` is a direction in space.

It says:

> “This is the axis along which Math and Writing differ.”

So when you train a linear probe, you are discovering a semantic axis.

Example:

```text
math direction  ←──────────────→ writing direction
```

A point’s position along that axis tells the probe which class it belongs to.

This is one of the most important interpretability ideas in modern ML:

> Concepts can appear as directions in activation space.

---

# 5. Why This Matters for LLMs

In real language models, you can take hidden states from inside the model and train probes for things like:

```text
truth vs. falsehood
safe vs. unsafe
question vs. answer
math vs. prose
code vs. natural language
positive vs. negative sentiment
```

If a linear probe succeeds, it suggests the model’s internal vectors already contain that information in a readable geometric form.

The probe does not create the knowledge.

It detects that the knowledge is already there.

That is the key distinction.

---

# 6. Why Regularization Matters

The revised script adds L2 weight decay.

That is important.

Without regularization, a probe can sometimes cheat by finding accidental separations in noisy high-dimensional data.

L2 regularization penalizes large weights:

```text
loss = classification_error + λ * sum(w²)
```

This encourages the probe to find a simpler separating direction.

The intuition:

> Prefer the cleanest, smallest explanation that separates the data.

This matters because high-dimensional spaces can make separation surprisingly easy, even when the pattern is not meaningful.

---

# 7. Training Loop Intuition

The script trains the probe like this:

```text
start with random w and b
↓
make predictions
↓
measure error
↓
compute gradients
↓
adjust w and b
↓
repeat
```

Each update slightly rotates or shifts the separating boundary.

At the beginning, the line is random.

After training, it aligns with the true class separation.

---

# 8. Validation: Why Unseen Data Matters

The script does not only test on the training data.

It generates a separate validation set using another seed.

That matters because memorizing the training points is not enough.

The real question is:

> Did the probe learn the actual geometry, or just memorize the examples?

That is why validation accuracy is reported on unseen data.

The revised version also reports a Wilson confidence interval instead of a vague pass/fail threshold. That is a more statistically honest way to say:

> “Given this validation sample, here is the likely range of the true accuracy.”

---

# 9. The Main Lesson

A linear probe is a diagnostic instrument.

It asks:

```text
Is this concept linearly available inside the representation space?
```

If yes, then a simple dot product can recover it.

That means the model has organized the concept into a usable direction.

---

# 10. What to Run

Run:

```bash
elixir 05_linear_probe.exs
```

Watch for:

```text
Training Loss
Validation Accuracy
Final Weights
Final Bias
Wilson 95% CI
```

The most important thing to inspect is not just the accuracy.

Look at the final weights.

Those weights are the learned semantic direction.

---

# 11. Practice Questions

After running the script, answer:

1. What are the two synthetic classes?
2. What does the weight vector `w` represent geometrically?
3. Why is the dot product used in the classifier?
4. What does the bias `b` do?
5. Why do we need validation data?
6. Why can high-dimensional data be dangerously easy to separate?
7. What does L2 regularization discourage?

---

# 12. Minimum Vocabulary Before Course 3

| Term                 | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| Linear probe         | Simple classifier trained on frozen representations |
| Hyperplane           | A separating boundary in vector space               |
| Weight vector        | Direction that defines the classifier               |
| Bias                 | Offset that shifts the boundary                     |
| Sigmoid              | Converts a score into a probability                 |
| Binary cross entropy | Loss function for two-class classification          |
| Gradient descent     | Iterative parameter improvement                     |
| Regularization       | Penalty that discourages overfitting                |
| L2 weight decay      | Penalizes large squared weights                     |
| Validation set       | Unseen data used to test generalization             |
| Semantic axis        | Direction in representation space tied to meaning   |

---

# One-Sentence Summary

**Crash Course 2 teaches that if a simple linear probe can separate concepts in vector space, then the model’s internal representations already contain those concepts as readable geometric directions.**
