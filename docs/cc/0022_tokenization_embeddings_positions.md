# Foundation C: Tokenization, Embeddings, and Positions

This foundation should sit after:

```text
Foundation A: Shape Literacy
Foundation B: Gradients, Loss, and Backprop
```

and before:

```text
Foundation D: The Full Transformer Block
```

The goal:

> Understand how raw text becomes vectors, and why Transformers need positional information before attention can work correctly.

So far we have talked about token vectors as if they already exist. This lesson fills in the missing bridge:

```text
text
→ tokens
→ token IDs
→ embedding vectors
→ position-aware vectors
→ Transformer layers
```

Your existing lesson docs already gesture toward positional mechanisms like RoPE in the extended material, but the core crash-course path needs this explicit foundation before deeper Transformer architecture. 

---

# 1. The problem: neural nets do not read text

A neural network does not directly understand:

```text
"The cat sat"
```

It understands numbers.

So the first problem is:

> How do we turn text into numbers without losing too much structure?

The answer is:

```text
tokenization + embeddings
```

Tokenization turns text into discrete pieces.
Embeddings turn those pieces into vectors.

---

# 2. What is a token?

A **token** is a chunk of text that the model treats as one unit.

A token might be:

```text
a word
part of a word
punctuation
a space-like marker
a special symbol
```

Example:

```text
"The cat sat"
```

might become:

```text
["The", " cat", " sat"]
```

But another sentence:

```text
"unbelievable"
```

might become:

```text
["un", "believ", "able"]
```

The model does not necessarily use whole words.

Modern LLMs usually use **subword tokens**.

---

# 3. Why subword tokens?

Whole-word tokenization has problems.

Suppose the model sees:

```text
cat
cats
catlike
catastrophic
```

If every word is separate, the vocabulary explodes.

Subword tokenization lets the model reuse pieces:

```text
cat
s
like
astro
phic
```

This helps with:

```text
rare words
new words
misspellings
names
code
multiple languages
```

So tokenization is a compromise:

> break text into reusable chunks that are smaller than full words but larger than characters.

---

# 4. Token IDs

After tokenization, each token gets an integer ID.

Example:

```text
"The cat sat"
```

could become:

```text
["The", " cat", " sat"]
```

then:

```text
[791, 8415, 10139]
```

The exact numbers depend on the tokenizer.

The important thing:

```text
tokens are text chunks
token IDs are integer labels for those chunks
```

But token IDs are not meaningful by themselves.

ID `8415` is not “larger” or “more semantic” than ID `791`.

They are just lookup numbers.

---

# 5. The vocabulary

The model has a fixed vocabulary:

```text
token ID 0      → some token
token ID 1      → some token
token ID 2      → some token
...
token ID 50000  → some token
```

The vocabulary is like a giant dictionary:

```text
token string ↔ token ID
```

During preprocessing:

```text
text → token IDs
```

During generation:

```text
token IDs → text
```

---

# 6. The embedding table

Token IDs are discrete.

But neural networks need vectors.

So the model has an **embedding table**.

Think of it as a matrix:

```text
embedding_table shape: {vocab_size, hidden_dim}
```

Example:

```text
{50,000, 768}
```

Meaning:

```text
50,000 possible tokens
768 numbers per token vector
```

Each row is the learned vector for one token.

So if:

```text
token ID = 8415
```

the model looks up:

```text
embedding_table[8415]
```

and gets:

```text
a 768-dimensional vector
```

This is the token embedding.

---

# 7. Embedding lookup is not a hashmap, but it feels like one

Conceptually, this part is closer to a lookup table:

```text
token ID → vector
```

But the vector is learned.

The model does not merely store a static dictionary definition.

It learns coordinates useful for prediction.

So:

```text
"cat" vector
```

may end up near vectors for:

```text
kitten
dog
animal
pet
fur
meow
```

because those tokens appear in related contexts during training.

The embedding table is the first learned representation layer.

---

# 8. Shape after embedding

Suppose we tokenize:

```text
"The cat sat"
```

into 3 tokens.

Each token gets a vector of size 4 in the toy scripts.

Then:

```text
token IDs shape: {seq=3}
```

becomes:

```text
embeddings shape: {seq=3, dim=4}
```

Example:

```text
"The" → [1.0, 0.5, 0.2, 0.1]
"cat" → [0.1, 2.0, 0.8, 0.2]
"sat" → [0.2, 0.1, 1.5, 1.8]
```

So the whole sentence becomes:

```text
X shape: {seq=3, dim=4}
```

This is exactly the kind of shape used in the self-attention script: three token vectors, each with four features. 

---

# 9. The missing problem: attention has no built-in order

Self-attention compares every token to every other token.

That is powerful.

But raw attention by itself does not know order.

Without position information, these contain the same tokens:

```text
"cat sat on mat"
"mat sat on cat"
```

The token set is similar, but the meaning is different.

Order matters.

So the model needs some way to know:

```text
this is token 0
this is token 1
this is token 2
...
```

That is the job of positional information.

---

# 10. Why position is not optional

Attention computes relationships like:

```text
QKᵀ
```

That tells us:

```text
which tokens relate to which tokens
```

But if the token vectors contain no position information, attention sees a bag of vectors.

It can ask:

```text
which token matches this token?
```

but not naturally:

```text
which token came before this token?
which token came after?
how far away is it?
```

Language depends heavily on order.

Examples:

```text
dog bites man
man bites dog
```

Same words. Different meaning.

So Transformers need token identity plus position.

---

# 11. Absolute positional embeddings

One simple method is to learn a position vector for each slot.

Example:

```text
position 0 → vector
position 1 → vector
position 2 → vector
```

Then add it to the token embedding:

```text
input_vector = token_embedding + position_embedding
```

So:

```text
"cat" at position 1
```

gets a different final vector than:

```text
"cat" at position 5
```

because the position vector is different.

The model sees:

```text
what token this is
+
where it is
```

---

# 12. Shape of positional embeddings

For a sequence:

```text
{seq=3, dim=4}
```

token embeddings have shape:

```text
{3, 4}
```

position embeddings also have shape:

```text
{3, 4}
```

Then:

```text
token_embeddings + position_embeddings
```

returns:

```text
{3, 4}
```

Same shape, richer meaning.

This preserves the standard Transformer input shape:

```text
{seq, hidden_dim}
```

or with batches:

```text
{batch, seq, hidden_dim}
```

---

# 13. Learned vs fixed positions

There are two broad styles.

## Learned positional embeddings

The model learns a vector for each position.

```text
position 0 has trainable vector
position 1 has trainable vector
position 2 has trainable vector
```

Benefit:

```text
simple and flexible
```

Weakness:

```text
may generalize poorly beyond trained context length
```

## Fixed sinusoidal positions

The original Transformer used sinusoidal position encodings.

These are deterministic wave patterns.

Benefit:

```text
positions have mathematical structure
```

Weakness:

```text
less adaptive than learned embeddings
```

---

# 14. RoPE: Rotary Position Embeddings

Many modern LLMs use **RoPE**, or Rotary Position Embeddings.

Instead of adding a position vector, RoPE rotates the Query and Key vectors depending on position.

The extended lesson docs describe RoPE as applying a position-dependent rotation to Query and Key vectors, and note that these rotations affect the geometry of the Key space. 

Simple intuition:

```text
position changes the angle of Q and K vectors
```

So relative positions become visible through dot products.

RoPE is elegant because it helps attention understand relative distance:

```text
token A is 1 step away
token B is 10 steps away
token C is 100 steps away
```

The position is baked into how Q and K align.

---

# 15. Why RoPE affects Q and K, not usually V

Attention matching happens through:

```text
Q · K
```

So position matters most during matching.

The model needs to know:

```text
should this query attend to that key, given their positions?
```

The Value is the content being retrieved.

So RoPE usually modifies the matching vectors:

```text
Q and K
```

not the payload vectors:

```text
V
```

This ties directly back to Q/K/V:

```text
Q and K = search/matching system
V = content being mixed
```

Position changes the search geometry.

---

# 16. Token embedding vs positional information

A token embedding answers:

```text
what token is this?
```

A position embedding answers:

```text
where is this token?
```

The Transformer input needs both:

```text
what + where
```

Example:

```text
"dog bites man"
```

The model needs:

```text
dog = token identity
position 0 = subject-like location
bites = token identity
position 1 = verb-like location
man = token identity
position 2 = object-like location
```

Without positions, it cannot reliably distinguish:

```text
dog bites man
```

from:

```text
man bites dog
```

---

# 17. Embeddings are learned coordinates, not definitions

An embedding is not a dictionary definition.

It is not:

```text
cat = "small furry animal"
```

It is more like:

```text
cat = coordinates useful for predicting surrounding text
```

During training, the model learns embeddings because they help reduce loss.

If similar tokens appear in similar contexts, their vectors tend to organize similarly.

That is why embeddings become semantic.

Meaning emerges from prediction pressure.

---

# 18. Embedding table training

During training, the embedding table receives gradients like other parameters.

If a token appears in a batch, its embedding row participates in the forward pass.

Then backprop updates that row to reduce loss.

So the embedding for `"cat"` changes over training because the model repeatedly learns:

```text
given contexts with "cat", predict better next tokens
```

Embeddings are not handcrafted.

They are learned.

---

# 19. Input embeddings and output logits

LLMs use embeddings at the input.

But generation also needs to choose an output token.

At the end, the model produces logits over the vocabulary:

```text
logits shape: {vocab_size}
```

or batched:

```text
{batch, vocab_size}
```

Each logit is a score for one possible next token.

Then:

```text
softmax(logits)
```

turns scores into probabilities.

Then the model chooses or samples the next token.

So the full loop is:

```text
token IDs
→ input embeddings
→ Transformer layers
→ output logits over vocabulary
→ next token ID
→ repeat
```

---

# 20. Autoregressive preview

Most LLMs generate one token at a time.

Example:

```text
Prompt: "The cat"
```

Model predicts:

```text
" sat"
```

Then the new sequence becomes:

```text
"The cat sat"
```

Then it predicts the next token:

```text
" on"
```

Then:

```text
"The cat sat on"
```

and so on.

This is called **autoregressive generation**.

We will return to this later, but tokenization and embeddings are the entry point.

---

# 21. Why tokenization affects model behavior

Tokenization is not neutral.

It affects:

```text
how many tokens a prompt uses
how code is split
how names are represented
how rare words are handled
how multilingual text is processed
how long the effective context is
```

Example:

```text
"ChatGPT"
```

could be one token in one tokenizer, several tokens in another.

A word split into many tokens costs more context and may be harder for the model to handle cleanly.

So tokenization is part of model behavior, not just preprocessing.

---

# 22. Token count vs word count

A token is not a word.

A rough English rule of thumb:

```text
1 token ≈ 3-4 characters
```

or:

```text
100 tokens ≈ 75 words
```

But this varies.

Code, math, JSON, and unusual names may tokenize differently.

This matters because model context windows are measured in tokens, not words.

---

# 23. Special tokens

Models also use special tokens.

Examples:

```text
beginning of text
end of text
user message marker
assistant message marker
padding token
unknown token
separator token
```

Chat models especially rely on special formatting tokens to distinguish roles:

```text
system
user
assistant
tool
```

So the model is not just reading your raw words. It is reading a formatted token sequence.

---

# 24. The shape story

For one sequence:

```text
raw text
```

becomes:

```text
tokens: ["The", " cat", " sat"]
```

then:

```text
token IDs: {seq=3}
```

then embedding lookup:

```text
token embeddings: {seq=3, hidden_dim=4}
```

then positional information:

```text
position-aware embeddings: {seq=3, hidden_dim=4}
```

then Transformer:

```text
hidden states: {seq=3, hidden_dim=4}
```

For real models:

```text
{batch, seq}
→ {batch, seq, hidden_dim}
→ {batch, seq, hidden_dim}
```

The shape stays stable through most Transformer layers.

The contents become richer.

---

# 25. The key mental model

Tokenization asks:

```text
what discrete pieces are in this text?
```

Embeddings ask:

```text
what vector should represent each piece?
```

Position encoding asks:

```text
where is each piece in the sequence?
```

Attention asks:

```text
which pieces should exchange information?
```

MLP layers ask:

```text
what nonlinear transformations should happen to each token’s current representation?
```

The Transformer stack repeatedly refines the token vectors.

---

# 26. Common confusions

## “Is a token the same as a word?”

No. A token can be a word, part of a word, punctuation, or formatting marker.

## “Is a token ID meaningful?”

Not by itself. It is just an index into the embedding table.

## “Is an embedding a definition?”

No. It is learned coordinates useful for prediction.

## “Does attention know order automatically?”

No. Position information must be added or encoded.

## “Why does RoPE rotate Q and K?”

Because Q/K matching is where positional relationships affect attention scores.

## “Does the shape change after embedding?”

Yes:

```text
token IDs: {seq}
embeddings: {seq, hidden_dim}
```

The model moves from discrete IDs to dense vectors.

---

# 27. Practice questions

Answer these before continuing:

1. What is a token?
2. Why do modern models use subword tokens?
3. What is a token ID?
4. Why is a token ID not semantically meaningful by itself?
5. What is an embedding table?
6. What shape does an embedding table usually have?
7. What does embedding lookup do?
8. Why does attention need positional information?
9. What is the difference between token identity and token position?
10. What problem does RoPE solve?
11. Why does RoPE affect Q and K more directly than V?
12. What is autoregressive generation?
13. Why does tokenization affect context length?
14. What is the shape transition from token IDs to embeddings?

---

# One-sentence summary

**Tokenization turns text into discrete IDs, embeddings turn those IDs into learned vectors, and positional information tells the Transformer where each vector sits in the sequence.**
