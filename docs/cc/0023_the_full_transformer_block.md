# Foundation D: The Full Transformer Block

This foundation should sit after:

```text
Foundation A: Shape Literacy
Foundation B: Gradients, Loss, and Backprop
Foundation C: Tokenization, Embeddings, and Positions
```

The goal:

> Understand what happens inside one Transformer layer, not just self-attention in isolation.

So far, we have studied pieces:

```text
tokens
→ embeddings
→ positions
→ Q/K/V attention
→ softmax scaling
→ value mixing
```

But a Transformer is not just attention.

A full Transformer block is usually:

```text
input hidden states
↓
normalization
↓
self-attention
↓
residual add
↓
normalization
↓
MLP / feed-forward network
↓
residual add
↓
output hidden states
```

The key idea:

> A Transformer block repeatedly lets tokens exchange information, then lets each token privately think through that information.

---

# 1. The Transformer starts with hidden states

After tokenization, embedding lookup, and positional information, we have hidden states:

```text
X shape: {seq, hidden_dim}
```

or with batching:

```text
X shape: {batch, seq, hidden_dim}
```

Each token has one vector.

Example:

```text
"The" → vector
"cat" → vector
"sat" → vector
```

At the beginning, these vectors mostly represent token identity plus position.

After many Transformer layers, they represent richer context:

```text
"The" → determiner connected to cat
"cat" → subject of sat
"sat" → action performed by cat
```

*(Note: These descriptions are interpretive glosses from probing research, not direct readouts of the hidden state content. The actual vector content is not human-interpretable without a trained probe.)*

The shape often stays the same:

```text
{seq, hidden_dim}
→ {seq, hidden_dim}
→ {seq, hidden_dim}
```

But the information inside the vectors becomes more contextual.

---

# 2. The two jobs inside a Transformer block

A Transformer block has two major sublayers:

```text
1. Attention
2. MLP / feed-forward network
```

They do different jobs.

## Attention

Attention lets tokens talk to each other.

It answers:

```text
Which other tokens should this token read from?
```

## MLP

The MLP lets each token process its own updated information.

It answers:

```text
Now that I have context, what features should I compute or transform?
```

So:

```text
attention = communication between tokens
MLP       = private computation inside each token
```

*(Note: This framing comes from mechanistic interpretability research (Elhage et al., 'A Mathematical Framework for Transformer Circuits', 2021) and is a useful model, not an architectural constraint — attention can also perform computation and MLPs can perform a form of pattern matching.)*

That distinction is crucial.

---

# 3. Why attention alone is not enough

Attention mixes information across tokens.

For `"The cat sat"`:

```text
cat can read from The and sat
sat can read from cat
The can read from cat
```

But after tokens exchange information, the model still needs to transform that information. After tokens exchange information, each token's vector contains a mixture of context. The MLP can then apply nonlinear transformations to this mixture — for example, detecting whether a specific combination of features is present and amplifying or suppressing it.

That is the MLP’s job.

Attention moves information around.

MLP processes information.

---

# 4. The residual stream

The **residual stream** is the main highway running through the model.

Each layer does not replace the hidden state completely.

Instead, it adds updates to it.

Conceptually:

```text
new_x = old_x + update
```

This additive structure means each layer's contribution can be analyzed independently — the residual stream at layer L equals the sum of the original embedding plus all previous layer outputs. Sublayers write updates back into the shared vector channel.

This is incredibly important.

A layer does not say:

```text
throw away the old representation
```

It says:

```text
keep the old representation
add something useful
```

That makes deep models trainable and lets different layers contribute different refinements.

---

# 5. Why residual connections matter

Without residual connections, each layer would have to preserve everything important while also adding new information.

That is hard.

With residuals:

```text
x₁ = x₀ + attention_update
x₂ = x₁ + mlp_update
```

Each sublayer can specialize in adding a correction.

Think of it like editing a document.

Bad workflow:

```text
rewrite the whole document from scratch each time
```

Residual workflow:

```text
keep the document
add edits in the margin
```

Residual connections also solve the vanishing gradient problem in deep networks: the gradient of the loss with respect to early layers has a direct additive path through the residual connections, bypassing the multiplicative chain that causes gradients to shrink.

That is how Transformer layers accumulate meaning.

---

# 6. The simplest Transformer block

A simplified block looks like:

```text
X
↓
Y = X + Attention(X)
↓
Z = Y + MLP(Y)
```

*(Note: MLP receives the attention-updated residual Y, not the original X.)*

But real modern Transformers usually add normalization around the sublayers.

The common modern pattern is **pre-norm**:

```text
X
↓
Norm
↓
Attention
↓
Residual Add
↓
Norm
↓
MLP
↓
Residual Add
```

In formula form:

```text
Y = X + Attention(Norm(X))
Z = Y + MLP(Norm(Y))
```

Output:

```text
Z
```

This `Z` becomes the input to the next Transformer block.

---

# 7. LayerNorm / RMSNorm

Normalization keeps activations numerically stable. As vectors move through many layers, their values can grow, shrink, or drift.

LayerNorm normalizes each token's hidden vector to have zero mean and unit variance across the hidden dimension, then applies a learned scale and shift. RMSNorm omits the mean subtraction (only normalizes by RMS). Both prevent the magnitude of hidden states from growing exponentially with depth.

Common normalization types:

* **LayerNorm:** `LayerNorm(x) = (x - μ) / (σ + ε) * γ + β` where `μ`, `σ` are computed over the hidden dimension of each token independently, and `γ`, `β` are learned parameters.
* **RMSNorm:** normalizes by the root-mean-square rather than computing full variance.

The concept is:

> Normalize hidden states so each layer receives inputs in a stable range.

This helps gradients flow and makes training more stable.

---

# 8. Pre-norm vs post-norm

There are two broad layouts.

## Post-norm

Older-style:

```text
Y = Norm(X + Attention(X))
Z = Norm(Y + MLP(Y))
```

The norm happens after the residual addition.

## Pre-norm

Common in modern LLMs:

```text
Y = X + Attention(Norm(X))
Z = Y + MLP(Norm(Y))
```

The norm happens before the sublayer.

Pre-norm often trains more stably in very deep Transformers. In post-norm, the residual addition happens inside the normalization, which can cause the gradient to be divided by the norm magnitude — leading to instability in deep networks. In pre-norm, the normalization is inside the residual branch, leaving a clean gradient path directly through the additive residual stream.

The exact architecture varies by model, but the core pattern remains:

```text
normalize
compute update
add update to residual stream
```

---

# 9. Self-attention inside the block

Inside the attention sublayer:

```text
X → Q, K, V
QKᵀ / sqrt(d_k)
softmax(...)
softmax(...)V
```

Shape story for one head:

```text
X:                 {seq, hidden_dim}
Q, K, V:            {seq, head_dim}
attention scores:   {seq, seq}
attention weights:  {seq, seq}
attention output:   {seq, head_dim}
```

*(Note: The projection matrices for one head have shape `{hidden_dim, head_dim}`, compressing the full hidden dimension down to the per-head dimension: `{seq, hidden_dim} × {hidden_dim, head_dim} → {seq, head_dim}`.)*

For multi-head attention, this happens in parallel across heads.

---

# 10. Multi-head attention

Real Transformers usually do not use one attention head.

They use many.

Example:

```text
hidden_dim = 768
num_heads = 12
head_dim = 64
```

Because:

```text
12 × 64 = 768
```

The hidden vector is split into multiple attention subspaces.

Shape:

```text
{batch, seq, hidden_dim}
→ {batch, heads, seq, head_dim}

*(Concretely: reshape `hidden_dim` → `(heads, head_dim)` giving `{batch, seq, heads, head_dim}`, then transpose axes 1 and 2 to get `{batch, heads, seq, head_dim}`. This puts the heads axis before the sequence axis so each head operates on the full sequence independently.)*
```

Each head can learn a different relationship pattern.

One head might track:

```text
subject ↔ verb
```

Another might track:

```text
previous matching word
```

Another might track:

```text
punctuation or phrase boundaries
```

Another might track:

```text
copying behavior
```

*(Note: These are empirical observations from interpretability research; the actual learned functions vary by model and training.)*

The outputs of all heads are concatenated back together.

```text
heads outputs: {batch, heads, seq, head_dim}
↓
concat
{batch, seq, hidden_dim}
```

Then a final output projection mixes the heads.

---

# 11. Attention output projection

After multi-head attention, the model usually applies an output matrix:

```text
O = concat(heads) W_o
```

Why?

Because each head computed information in its own subspace.

After each head produces `{batch, seq, head_dim}`, the heads are concatenated along the last axis: `concat({batch, seq, head_dim} × num_heads) → {batch, seq, hidden_dim}`. Then the output projection `W_o: {hidden_dim, hidden_dim}` mixes across heads.

Shape:

```text
concat heads: {batch, seq, hidden_dim}
W_o:          {hidden_dim, hidden_dim}
output:       {batch, seq, hidden_dim}
```

Then this output is added back to the residual stream:

```text
Y = X + attention_output
```

Same shape. Updated information.

---

# 12. The MLP / feed-forward sublayer

After attention, each token goes through an MLP.

The MLP is applied independently to each token vector. It does not mix tokens. Concretely: the MLP weight matrices have shapes `{hidden_dim, mlp_dim}` and `{mlp_dim, hidden_dim}`, with no sequence axis — the same transformation is applied to each token's `{hidden_dim}` vector independently (equivalent to a 1×1 convolution across the sequence).

For each token:

```text
hidden vector
↓
expand
↓
nonlinearity
↓
contract
```

Typical shape:

```text
{hidden_dim}
→ {mlp_dim}
→ {hidden_dim}
```

Example:

```text
768 → 3072 → 768
```

The MLP often expands the vector to a larger internal space, applies a nonlinearity, then projects back down.

---

# 13. Why the MLP expands

The expansion gives the model more room to compute features.

A common ratio is about 4x:

```text
hidden_dim = 768
mlp_dim = 3072
```

The expansion creates a larger intermediate space where the nonlinear activation can distinguish patterns that are entangled at `hidden_dim`. After the activation, the contraction `{mlp_dim} → {hidden_dim}` selects and compresses the detected patterns back into the residual stream.

Attention brought in context.

The MLP processes that context.

---

# 14. Why nonlinearities matter

If a model only used linear layers, then stacking many layers would still collapse into one big linear transformation. Formally, `W_n ··· W_2 W_1 = W_combined` — any stack of linear layers is exactly equivalent to a single linear layer with weights equal to their product. This **linear collapse** is provable by the associativity of matrix multiplication.

Example:

```text
linear(linear(linear(x))) = another linear(x)
```

Nonlinear activation functions prevent this collapse.

Common activations:

* **ReLU:** applies `max(0, x)`.
* **GELU:** applies a smooth approximation of ReLU that allows small negative values.
* **SwiGLU / GeGLU:** *gated* activations that multiply two linear projections together, effectively allowing the network to gate which features to pass through. SwiGLU is used in Llama, Mistral, and most modern LLMs.

The activation lets the model create conditional behavior:

```text
if this feature is present, amplify it
if not, suppress it
```

This is where a lot of feature computation happens.

---

# 15. MLP as feature transformer

A useful mental model:

```text
attention decides what information arrives
MLP decides what to do with it
```

Example for token `"sat"`:

Attention might bring in information from `"cat"`.

Now the `"sat"` vector contains context:

```text
action + subject info
```

The MLP can transform that into features like:

```text
past-tense action
subject is animal
simple sentence structure
likely next token may be preposition
```

This is simplified, but directionally useful.

---

# 16. One full block in shape form

For one sequence:

```text
X {seq, hidden_dim}
```

Pre-norm attention:

```text
Norm(X)                         {seq, hidden_dim}
Attention(Norm(X))              {seq, hidden_dim}
Y = X + Attention(Norm(X))      {seq, hidden_dim}
```

Pre-norm MLP:

```text
Norm(Y)                         {seq, hidden_dim}
MLP(Norm(Y))                    {seq, hidden_dim}
Z = Y + MLP(Norm(Y))            {seq, hidden_dim}
```

Output:

```text
Z {seq, hidden_dim}
```

The shape stays stable so many blocks can be stacked. During training, these same shapes flow *backward* through the block — gradients have the same shapes as their corresponding forward activations.

---

# 17. Stacking Transformer blocks

A model has many blocks.

Example:

```text
Block 1
Block 2
Block 3
...
Block 32
```

Each block refines the hidden states.

Early layers may learn lower-level patterns:

```text
token identity
position
local syntax
```

Middle layers may learn:

```text
phrases
entities
relations
induction/copying
```

Later layers may learn:

```text
task intent
answer structure
next-token decision features
```

This is an oversimplification, but useful.

The residual stream carries information through all layers.

---

# 18. Decoder-only Transformer

Most chat LLMs are **decoder-only Transformers**.

That means they predict the next token autoregressively.

They use **causal self-attention**.

Causal means:

```text
a token can attend to previous tokens
but not future tokens
```

When predicting token 5, the model cannot peek at token 6.

So the attention matrix uses a causal mask.

---

# 19. Causal mask

Without a mask, every token can attend to every token:

```text
      K0 K1 K2 K3
Q0    ✓  ✓  ✓  ✓
Q1    ✓  ✓  ✓  ✓
Q2    ✓  ✓  ✓  ✓
Q3    ✓  ✓  ✓  ✓
```

With a causal mask:

```text
      K0 K1 K2 K3
Q0    ✓  ×  ×  ×
Q1    ✓  ✓  ×  ×
Q2    ✓  ✓  ✓  ×
Q3    ✓  ✓  ✓  ✓
```

In practice, masked positions are set to `-∞` (or a large negative value like `-1e9`) before the softmax. Since `softmax(-∞) = 0`, these positions contribute nothing to the output.

Each token can only attend to positions with equal or lower index — i.e., earlier in the sequence as defined by token order, not by linguistic direction.

This preserves the next-token prediction task.

---

# 20. Encoder vs decoder intuition

Classic Transformers had encoders and decoders.

## Encoder

Reads the whole input.

Useful for:

```text
classification
embedding
bidirectional understanding
```

Encoder (bidirectional) attention: each token can attend to all other tokens in both directions.

## Decoder

Generates output one token at a time.

Useful for:

```text
language modeling
chat
code generation
completion
```

Decoder attention is causal.

Modern GPT-style LLMs are decoder-only.

---

# 21. From final hidden state to next token

After the final Transformer block, we have:

```text
hidden states: {seq, hidden_dim}
```

To predict the next token, the model looks at the last token’s hidden state:

```text
last hidden vector: {hidden_dim}
```

*(Note: For decoder-only models with causal masking, only the final position's hidden state has information about the full input — earlier positions haven't 'seen' later tokens. This is why the final position is used for next-token prediction.)*

Then projects it to vocabulary logits:

```text
logits = hidden W_vocab
```

Shape:

```text
{hidden_dim} × {hidden_dim, vocab_size}
→ {vocab_size}
```

Each logit is a score for one possible next token.

Then:

```text
softmax(logits)
```

turns scores into probabilities.

The model samples or chooses the next token.

---

# 22. Autoregressive loop

The generation loop:

```text
prompt tokens
↓
Transformer forward pass
↓
next-token logits
↓
choose next token
↓
append token to prompt
↓
repeat
```

Example:

```text
"The cat"
→ predicts " sat"
"The cat sat"
→ predicts " on"
"The cat sat on"
→ predicts " the"
```

This is inference.

During training, the model learns this by predicting next tokens over massive text datasets.

---

# 23. Training objective for decoder-only LLMs

The model sees a sequence:

```text
The cat sat on the mat
```

Training asks it to predict each next token:

```text
input:  The       target: cat
input:  The cat   target: sat
input:  The cat sat target: on
...
```

The causal mask allows all next-token predictions to be computed in a single forward pass: position `i` predicts token `i+1` using only positions `0..i`. Without this parallelism, training would require `T` sequential forward passes per sequence, making it completely impractical.

The loss is usually cross entropy over the vocabulary.

So the model is trained to assign high probability to the correct next token.

---

# 24. Where Q/K/V lives in the full block

Q/K/V is not the whole model.

It is inside the attention sublayer:

```text
Transformer block
├── Norm
├── Self-attention
│   ├── Q projection
│   ├── K projection
│   ├── V projection
│   ├── QKᵀ / sqrt(d)
│   ├── softmax
│   └── weighted V mix
├── Residual add
├── Norm
├── MLP
└── Residual add
```

So Q/K/V is one mechanism inside a larger repeated architecture. The projection matrices `Wq`, `Wk`, `Wv`, `Wo` are the trainable parameters of the attention sublayer; they are the primary targets for LoRA adaptation.

---

# 25. Why the block preserves shape

The block must preserve:

```text
{seq, hidden_dim}
```

because the next block expects the same shape.

So both attention and MLP return updates shaped like the residual stream:

```text
attention_output {seq, hidden_dim}
mlp_output       {seq, hidden_dim}
```

Then residual additions are valid:

```text
X + attention_output
Y + mlp_output
```

Shape preservation is what lets you stack many blocks. This shape constraint means you can add or remove layers without changing the interface — a 12-layer and a 32-layer model with the same `hidden_dim` are architecturally compatible at each layer boundary. It also means LoRA adapters trained for one layer depth can be applied at any layer.

---

# 26. The Transformer block as a communication-computation cycle

The best mental model:

```text
attention = communicate
MLP = compute
residual = remember
norm = stabilize
```

Each block does:

```text
stabilize
communicate
remember
stabilize
compute
remember
```

Then repeats.

That is the full rhythm of a Transformer.

---

# 27. What changes across layers?

The vectors keep the same shape, but their meaning changes.

Layer 0:

```text
mostly token + position
```

Layer 5:

```text
local syntax and phrase information
```

Layer 15:

```text
longer-range dependencies and task context
```

Layer 31:

```text
features useful for next-token prediction
```

Each layer writes more information into the residual stream. These are rough empirical patterns observed in BERT-scale and GPT-scale models; exact layer assignments vary by model size, training data, and task. Smaller models compress all processing into fewer layers.

---

# 28. Important implementation variants

Different models vary in details:

```text
LayerNorm vs RMSNorm
GELU vs SwiGLU
absolute positions vs RoPE
multi-head attention vs grouped-query attention
dense MLP vs MoE MLP (MoE replaces the dense MLP with a sparse mixture of expert MLPs, see lesson 6; the attention sublayer is unchanged)
post-norm vs pre-norm
```

But the core block pattern remains:

```text
attention update
residual add
MLP update
residual add
```

---

# 29. How this connects to previous lessons

## Shape Literacy

The block works because every sublayer returns compatible shapes:

```text
{batch, seq, hidden_dim}
```

## Gradients / Backprop

During training, loss gradients flow backward through:

```text
logits
→ final hidden state
→ Transformer blocks
→ attention/MLP parameters
→ embeddings
```

## Tokenization / Embeddings / Positions

The block does not receive raw text. It receives position-aware vectors.

## Q/K/V

Q/K/V is the attention subroutine inside the block.

## Softmax scaling

Scaling keeps Q/K matching trainable.

## LoRA

LoRA can be attached to projection matrices inside attention or MLP layers.

Common LoRA targets include:

```text
Wq
Wk
Wv
Wo
MLP projection matrices
```

LoRA works because these are just learned matrix transformations inside the block.

---

# 30. Minimal pseudocode

A simplified decoder Transformer block:

```text
def transformer_block(x):
    # x: {batch, seq, hidden_dim}

    a = norm1(x)
    a = causal_self_attention(a)
    x = x + a

    m = norm2(x)
    m = mlp(m)
    x = x + m

    return x
```

Causal self-attention:

```text
# Pseudocode (Python-style notation): @ denotes matrix multiply, .T denotes transpose.
# See 06_self_attention.exs for Elixir/Nx equivalent.
def causal_self_attention(x):
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    scores = (q @ k.T) / sqrt(head_dim) # Nx equivalent: Nx.dot(q, [1], k, [1])
    scores = apply_causal_mask(scores)

    weights = softmax(scores)
    out = weights @ v

    return out @ Wo
```

MLP:

```text
def mlp(x):
    h = activation(x @ W_up)
    return h @ W_down
```

That is the Transformer block in its simplest useful form.

---

# 31. Practice questions

Answer these before moving on:

1. In plain English, what does one Transformer block do? (State your current mental model before reading/revisiting the section, then compare with your post-reading understanding.)
2. What are the two main sublayers inside a Transformer block?
3. What job does attention perform?
4. What job does the MLP perform?
5. What is the residual stream?
6. Why do residual connections help?
7. What does normalization do?
8. What is the difference between pre-norm and post-norm?
9. Why does the block preserve `{seq, hidden_dim}`?
10. What is multi-head attention?
11. Why does attention need a causal mask in decoder-only LLMs?
12. How does the final hidden state become next-token logits?
13. What is the autoregressive generation loop?
14. Where do Q/K/V live inside the full block?
15. What happens mathematically if you stack two linear layers with no activation between them? Why does this make deep networks useless without nonlinearities?

---

# One-sentence summary

**A Transformer block stabilizes the residual stream, lets tokens communicate through attention, lets each token compute through an MLP, and writes both updates back into the same hidden-state shape so many layers can be stacked.**
