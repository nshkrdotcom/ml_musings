Based on the scripts and lesson docs, I’d place the learner at **early-intermediate programming level with strong conceptual curiosity**: comfortable reading Elixir scripts, ready for tensors/Nx, but still needs grounding in “why this matters” before jumping into attention, LoRA, MoE, and optimization. The best first crash course should therefore teach **Elixir → tensors → compiled numerical thinking → geometric intuition** before touching full ML architecture. 

# Introductory Crash Course 1: From Elixir Lists to ML Geometry

## Goal

By the end of this crash course, you should understand:

1. Why normal Elixir lists are not enough for machine learning.
2. What tensors are and why they matter.
3. Why compiled tensor math is different from ordinary interpreted code.
4. What a dot product means geometrically.
5. Why “direction” matters more than raw magnitude in embeddings.

This course maps directly to:

* `01_list_math.exs`
* `02_tensor_math.exs`
* `03_compiler.exs`
* `04_dot_product.exs`
* `quasi_orthogonality.exs`
* `hoeffding_bound.exs`
* `lesson_1_notes.txt`

---

# Part 1 — The Big Shift: Lists Are for Programs, Tensors Are for Models

In normal Elixir, a list like this:

```elixir
[1.0, 2.0, 3.0, 4.0]
```

is great for many programming tasks. But it is not how modern AI systems want to process numbers.

A standard Elixir list is a linked structure. Each item points to the next item. That is flexible, but it means the CPU has to hop around memory. This is called **pointer chasing**.

Machine learning wants something different:

```elixir
Nx.tensor([1.0, 2.0, 3.0, 4.0])
```

A tensor stores numbers in a compact numerical structure designed for fast math. Instead of thinking “loop over items,” ML systems think:

> Apply one mathematical operation across the entire block of numbers.

That is the first conceptual upgrade.

Normal programming says:

> “Do this operation one item at a time.”

Numerical ML programming says:

> “Treat the whole vector/matrix as one mathematical object.”

---

# Part 2 — Why Tensors Alone Are Not Magic

A beginner mistake would be thinking:

> “I used tensors, so it must be fast now.”

The scripts correctly teach that this is not automatically true.

There are three levels:

## Level 1: Elixir lists

Readable, familiar, but slow for large numerical work.

## Level 2: Interpreted tensors

Tensors on the wrong device or called outside `defn` incur per-operation dispatch overhead and potential host↔device copies.

## Level 3: Compiled tensors

This is where serious ML performance begins.

In Nx, `defn` marks a numerical function that can be compiled:

```elixir
defn multiply(a, b) do
  Nx.multiply(a, b)
end
```

This is not just “an Elixir function.” It becomes a numerical graph that a backend like EXLA/XLA can optimize.

The mental model is:

> `defn` turns your math into a compiled numerical recipe.

That recipe can then run closer to native hardware speed.

---

# Part 3 — The Most Important First ML Object: The Vector

A vector is just a list of numbers with geometric meaning:

```text
[0.0, 1.0]
```

That can mean “up” in 2D space.

```text
[1.0, 0.0]
```

That can mean “right.”

In ML, vectors often represent meaning:

```text
"math task"    → [2.1, 1.9, 0.4, ...]
"writing task" → [-1.8, -2.2, 0.1, ...]
```

These are illustrative; in practice the vector might represent a single token, a sentence, or an internal activation — the geometry is the same regardless of granularity.

An embedding is a vector that stores a concept as coordinates.

The first big idea is:

> Machine learning models turn concepts into directions in space.

---

# Part 4 — Dot Product: The “Are These Pointing the Same Way?” Test

The dot product is the core operation behind similarity, attention, routing, search, and probing.

For two vectors:

```text
u = [1, 0]
v = [0, 1]
```

Their dot product is:

```text
u · v = 1*0 + 0*1 = 0
```

That means they are perpendicular.

For:

```text
u = [1, 0]
v = [1, 0]
```

The dot product is:

```text
u · v = 1*1 + 0*0 = 1
```

That means they point the same way.

So the dot product answers:

> How much does one vector point in the direction of another?

That is why it appears everywhere in the later lessons.

Attention uses dot products.

Linear probes use dot products.

Routers use dot products.

Similarity search uses dot products.

MoE gating uses dot products.

The dot product is the “alignment detector” of neural networks.

---

# Part 5 — Normalization: Ignore Size, Keep Direction

Sometimes a vector is large only because it has bigger numbers, not because it means something different.

Example:

```text
[1, 1]
[100, 100]
```

These point in the same direction, but one is much longer.

For semantic similarity, direction often matters more than length. So we normalize vectors to length 1.

That means we project them onto a unit circle or unit sphere.

When both vectors are normalized to unit length, their dot product equals their cosine similarity — the standard formula `(u·v)/(||u|| ||v||)` reduces to just `u·v`.

Cosine similarity focuses on angle:

```text
same direction       →  1.0
perpendicular        →  0.0
opposite direction   → -1.0
```

This is why `04_dot_product.exs` is such an important bridge lesson. It turns raw arithmetic into geometric intuition.

---

# Part 6 — Why High Dimensions Are Weird and Powerful

In 2D or 3D, vectors easily overlap.

But in 4,096 or 8,192 dimensions, random vectors are almost always nearly perpendicular.

This is called **quasi-orthogonality**.

That sounds abstract, but the intuition is simple:

> In high dimensions, the volume of a hypersphere concentrates near its equatorial band relative to any reference direction. Any random vector is almost certainly near the equator of any other — i.e., near-perpendicular.

This is one reason large models can store many concepts inside the same residual stream.

A model can represent things like:

```text
plurality
past tense
math reasoning
code syntax
emotional tone
routing preference
```

as different directions in a huge vector space.

Because most directions are almost perpendicular, they can coexist without completely destroying each other.

This leads into the later idea of **superposition**:

> A neural network can pack many more useful features into a vector space than the raw number of dimensions might suggest. Formally, superposition occurs when a model represents more features than it has dimensions by tolerating small interference between nearly-orthogonal feature directions.

---

# Part 7 — The Core Mental Model

Here is the first crash-course mental model:

```text
Elixir list
  ↓
tensor
  ↓
compiled tensor operation
  ↓
vector geometry
  ↓
dot product similarity
  ↓
high-dimensional representation space
  ↓
modern neural network behavior
```

The learner should not rush into attention or LoRA yet. First, they need to deeply internalize this:

> Modern AI is mostly geometry plus compiled tensor math.

The scripts are teaching exactly that path.

---

# First Practice Assignment

Run these in order:

```bash
elixir 01_list_math.exs
elixir 02_tensor_math.exs
elixir 03_compiler.exs
elixir 04_dot_product.exs
elixir quasi_orthogonality.exs
elixir hoeffding_bound.exs
```

While running them, answer these questions in your own words:

1. Why are Elixir lists slower for large numerical workloads?
2. Why can interpreted tensors still be slow?
3. What does `defn` change?
4. What does a dot product measure?
5. Why do we normalize vectors?
6. Why are random vectors almost perpendicular in high dimensions?
7. Why does that matter for LLMs?

---

# Minimum Vocabulary to Master Before Course 2

You are ready for the next crash course when these words feel familiar:

| Term              | Meaning                                                          |
| ----------------- | ---------------------------------------------------------------- |
| Tensor            | A numerical array used for vectorized math                       |
| Backend           | The system that executes tensor operations                       |
| EXLA/XLA          | Compiler backend for optimized numerical execution               |
| `defn`            | Nx numerical function that can be compiled                       |
| Vector            | A coordinate representation of magnitude and direction           |
| Dot product       | Measures directional alignment                                   |
| Cosine similarity | Dot product after normalization                                  |
| Embedding         | A learned vector representation of meaning                       |
| Orthogonal        | Perpendicular; zero directional overlap                          |
| Quasi-orthogonal  | Almost perpendicular, common in high dimensions (dot product of two random unit vectors in D dimensions has std ≈ 1/√D) |
| Superposition     | Many features sharing one vector space with limited interference, enabled by quasi-orthogonality; interference scales as 1/√D per additional feature |

---

# The One-Sentence Summary

**Crash Course 1 teaches that AI systems work by turning meaning into high-dimensional vectors, then using compiled tensor math and dot products to measure, route, and transform those meanings efficiently.**
