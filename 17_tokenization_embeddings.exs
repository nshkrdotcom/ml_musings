# ===========================================================================
# LESSON 3d: Tokenization, Embeddings, and the Geometry of Position
# ===========================================================================
# This script bridges the gap between raw text preprocessing and Transformer
# input spaces. It demonstrates:
#   - Toy vocabulary construction and integer token ID mapping
#   - High-dimensional Embedding lookup ({seq} -> {seq, hidden_dim})
#   - Inherent permutation-equivariance of self-attention
#   - Incompatibility of sequence-blind attention with language structure
#   - How Positional Embeddings (Sinusoidal & Learned) break permutation-equivariance
#     to inject sequence structure and token order.

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule Tokenizer do
  @doc """
  A simple whitespace tokenizer mapping a toy vocabulary to unique integer IDs.
  """
  def toy_vocabulary do
    [
      "<pad>", "The", "cat", "sat", "on", "mat", "dog", "bites",
      "man", "a", "is", "happy", "furry", "friendly", "brown",
      "and", "the", "with", "sleeping", "<eos>"
    ]
  end

  def token_to_id_map do
    toy_vocabulary() |> Enum.with_index() |> Map.new()
  end

  def id_to_token_map do
    toy_vocabulary() |> Enum.with_index() |> Enum.into(%{}, fn {word, idx} -> {idx, word} end)
  end

  def tokenize(text) do
    String.split(text)
  end

  def encode(text) do
    map = token_to_id_map()
    text
    |> tokenize()
    |> Enum.map(fn word ->
      Map.get(map, word) || raise "Word '#{word}' not in toy vocabulary!"
    end)
  end

  def decode(ids) do
    map = id_to_token_map()
    ids
    |> Enum.map(fn id -> Map.fetch!(map, id) end)
    |> Enum.join(" ")
  end
end

defmodule EmbeddingMath do
  import Nx.Defn

  # Generate Sinusoidal Positional Embeddings
  # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  defn sinusoidal_positional_embeddings(opts \\ []) do
    opts = keyword!(opts, [:seq_len, :hidden_dim])
    seq_len = opts[:seq_len]
    hidden_dim = opts[:hidden_dim]

    # pos shape: {seq_len, 1}
    pos = Nx.reshape(Nx.iota({seq_len}), {seq_len, 1})
    
    # 2i shape: {1, half_dim}
    half_dim = div(hidden_dim, 2)
    i = Nx.reshape(Nx.iota({half_dim}), {1, half_dim})
    
    # divisor = 10000 ^ (2i / d_model)
    # hidden_dim is an integer keyword arg; Nx.as_type wraps it as a scalar tensor for the division.
    power = Nx.divide(Nx.multiply(2.0, i), Nx.as_type(hidden_dim, {:f, 32}))
    divisor = Nx.pow(10000.0, power)
    
    # angles shape: {seq_len, half_dim}
    angles = Nx.divide(pos, divisor)
    
    sin_vals = Nx.sin(angles)
    cos_vals = Nx.cos(angles)
    
    # Interleave sin and cos values along the hidden dimension
    Nx.concatenate([sin_vals, cos_vals], axis: 1)
  end
end

defmodule SimpleSelfAttention do
  import Nx.Defn

  # Stable softmax along the last dimension
  defn stable_softmax(t) do
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    t_shifted = Nx.subtract(t, max_vals)
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    Nx.divide(exps, sum_exps)
  end

  # Bidirectional raw self-attention (no batch axis for clarity)
  # x: {seq, hidden_dim}
  defn attention_forward(x, w_q, w_k, w_v, head_dim) do
    # 1. Projection
    queries = Nx.dot(x, [1], w_q, [0]) # {seq, head_dim}
    keys    = Nx.dot(x, [1], w_k, [0]) # {seq, head_dim}
    values  = Nx.dot(x, [1], w_v, [0]) # {seq, head_dim}

    # 2. Matchmaking (Q · K^T)
    raw_scores = Nx.dot(queries, [1], keys, [1]) # {seq, seq}

    # 3. Scale and Softmax
    scale_factor = Nx.sqrt(head_dim)
    scaled_scores = Nx.divide(raw_scores, scale_factor)
    attention_weights = stable_softmax(scaled_scores) # {seq, seq}

    # 4. Value extraction
    output = Nx.dot(attention_weights, [1], values, [0]) # {seq, head_dim}

    {output, attention_weights}
  end
end

# --- RUNNING THE TUTORIAL & EXPERIMENTS ---

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 3d: TOKENIZATION, EMBEDDINGS, AND THE GEOMETRY OF POSITION")
IO.puts(String.duplicate("=", 75))

# 1. TOY VOCABULARY AND ENCODING
prompt = "The cat sat on the mat"
token_ids = Tokenizer.encode(prompt)
# Assert encoded IDs are correct to sanity-check vocabulary alignment
if token_ids != [1, 2, 3, 4, 16, 5], do: raise "Encoded IDs mismatch! Got: #{inspect(token_ids)}"

IO.puts("STEP 1: TOY VOCABULARY AND PREPROCESSING")
IO.puts("  - Input Raw Sentence: \"#{prompt}\"")
IO.puts("  - Whitespace Tokens:  #{inspect(Tokenizer.tokenize(prompt))}")
IO.puts("  - Vocabulary size:    #{length(Tokenizer.toy_vocabulary())} words")
IO.puts("  - Encoded Integer IDs: #{inspect(token_ids)}")
IO.puts("  - Decoded Validation: \"#{Tokenizer.decode(token_ids)}\"\n")

# 2. EMBEDDING LOOKUP
hidden_dim = 6
vocab_size = length(Tokenizer.toy_vocabulary())
ids_tensor = Nx.tensor(token_ids) # shape {seq_len=6}

# Construct deterministic Embedding Table (vocab_size x hidden_dim = 20 x 6)
# Each word gets a distinct, frozen coordinates row.
# We use sin(iota) to keep magnitudes balanced and avoid any single token dominating.
indices = Nx.iota({vocab_size, hidden_dim})
embedding_table = Nx.sin(indices) |> Nx.multiply(0.5)

# Take vectors corresponding to our sequence
token_embeds = Nx.take(embedding_table, ids_tensor) # shape: {seq_len, hidden_dim}

IO.puts("STEP 2: HIGH-DIMENSIONAL EMBEDDING TABLE INDEX LOOKUP")
IO.puts("  - Token IDs Shape:               #{inspect(Nx.shape(ids_tensor))}")
IO.puts("  - Embedding Table Shape:        #{inspect(Nx.shape(embedding_table))}")
IO.puts("  - Looked-up Token Embeddings:   #{inspect(Nx.shape(token_embeds))}")
IO.puts("  - Representational Coordinates for [\"The\", \"cat\"]:")
IO.inspect(token_embeds[0..1])
IO.puts("")

# 3. GENERATING POSITIONAL EMBEDDINGS
seq_len = length(token_ids)
pe_sinusoidal = EmbeddingMath.sinusoidal_positional_embeddings(seq_len: seq_len, hidden_dim: hidden_dim)

# Construct learned absolute position table
learned_pos_table = Nx.broadcast(0.05, {10, hidden_dim})
pe_learned = Nx.slice(learned_pos_table, [0, 0], [seq_len, hidden_dim])

IO.puts("STEP 3: GENERATING POSITIONAL REPRESENTATIONS")
IO.puts("  - Sinusoidal PE Table Shape:     #{inspect(Nx.shape(pe_sinusoidal))}")
IO.puts("  - Sinusoidal PE Tensor Coordinates:")
IO.inspect(pe_sinusoidal)
IO.puts("  - Learned PE Slice Shape:        #{inspect(Nx.shape(pe_learned))}\n")

# 4. INHERENT PERMUTATION-EQUIVARIANCE OF RAW ATTENTION
# Construct non-uniform projection matrices to avoid degenerate attention where all projections align perfectly.
# We scale the iota coordinates to create non-trivial directions.
w_q = Nx.reshape(Nx.iota({hidden_dim, hidden_dim}), {hidden_dim, hidden_dim}) |> Nx.multiply(0.05)
w_k = Nx.reshape(Nx.iota({hidden_dim, hidden_dim}), {hidden_dim, hidden_dim}) |> Nx.multiply(0.03)
w_v = Nx.reshape(Nx.iota({hidden_dim, hidden_dim}), {hidden_dim, hidden_dim}) |> Nx.multiply(0.07)
head_dim = Nx.tensor(hidden_dim * 1.0)

# Compute raw attention output without position embeddings
{out_raw, _weights_raw} = SimpleSelfAttention.attention_forward(token_embeds, w_q, w_k, w_v, head_dim)

# Now, let's create a permuted sequence (swap "cat" at idx 1 and "sat" at idx 2)
# Original sequence tokens: ["The", "cat", "sat", "on", "the", "mat"]
# Permuted sequence tokens: ["The", "sat", "cat", "on", "the", "mat"]
permuted_ids = [1, 3, 2, 4, 16, 5] # Encoded IDs swapped
x_permuted = Nx.take(embedding_table, Nx.tensor(permuted_ids))

{out_permuted, weights_permuted} = SimpleSelfAttention.attention_forward(x_permuted, w_q, w_k, w_v, head_dim)

IO.puts("STEP 4: DEMONSTRATING THE PERMUTATION-EQUIVARIANCE OF RAW ATTENTION")
IO.puts("  * Raw self-attention is entirely sequence-blind. It operates purely as a set-to-set")
IO.puts("    relation map without native understanding of order.")
IO.puts("  * Let's swap Token 1 (\"cat\") and Token 2 (\"sat\") and inspect the output:")

# Verify that the output at idx 1 of the permuted sequence is exactly equal to the output
# at idx 2 of the original sequence, and vice versa!
diff_swap = Nx.subtract(out_permuted[1], out_raw[2])
l2_diff = Nx.sqrt(Nx.sum(Nx.pow(diff_swap, 2))) |> Nx.to_number()

IO.puts("  - L2 Difference between out_permuted[\"sat\"] and out_raw[\"sat\"]: #{Float.round(l2_diff, 8)}")
IO.puts("    (Should be exactly 0.0, indicating the vector swapped perfectly with the token!)")
IO.puts("  - Permuted Attention Weight Matrix:")
IO.inspect(weights_permuted)
IO.puts("")

# 5. HOW POSITIONAL EMBEDDINGS BREAK PERMUTATION-EQUIVARIANCE
# We inject order by adding position vectors to our token vectors.
# Let's add sinusoidal PE to the inputs.
x_with_pos = Nx.add(token_embeds, pe_sinusoidal)

# For the permuted sequence, the positional embeddings are added in sequence order (absolute index),
# meaning Token 1 gets PE 1, and Token 2 gets PE 2, despite them being different words now!
x_permuted_with_pos = Nx.add(x_permuted, pe_sinusoidal)

# Compute attention
{out_pos, _weights_pos} = SimpleSelfAttention.attention_forward(x_with_pos, w_q, w_k, w_v, head_dim)
{out_permuted_pos, _weights_permuted_pos} = SimpleSelfAttention.attention_forward(x_permuted_with_pos, w_q, w_k, w_v, head_dim)

# Check if permutation equivariance is now broken
diff_swap_pos = Nx.subtract(out_permuted_pos[1], out_raw[2])
l2_diff_pos = Nx.sqrt(Nx.sum(Nx.pow(diff_swap_pos, 2))) |> Nx.to_number()
IO.puts("  - L2 Difference between out_permuted_pos[\"sat\"] (pos 1) and out_raw[\"sat\"] (pos 2): #{Float.round(l2_diff_pos, 4)}")

# Check actual L2 difference between out_permuted_pos[1] and out_pos[2]
# If permutation-equivariance is broken, the vector at index 1 of the permuted output
# is NOT the same as index 2 of the original output, because the absolute positional context changed!
diff_actual_pos = Nx.subtract(out_permuted_pos[1], out_pos[2])
l2_diff_actual_pos = Nx.sqrt(Nx.sum(Nx.pow(diff_actual_pos, 2))) |> Nx.to_number()

IO.puts("STEP 5: HOW POSITIONAL EMBEDDINGS INJECT STRUCTURE (BREAKING EQUIVARIANCE)")
IO.puts("  * Adding positional embeddings injects a coordinate bias that varies by absolute position.")
IO.puts("  * Now, swapping \"cat\" and \"sat\" produces completely different contextual vectors because")
IO.puts("    their location relative to the rest of the sentence is structurally marked.")
IO.puts("  - L2 Difference between out_permuted_pos[\"sat\"] and out_pos[\"sat\"]: #{Float.round(l2_diff_actual_pos, 4)}")
IO.puts("    (Notice that this is non-zero! Permutation-equivariance is successfully broken!)")
IO.puts("  * The network now distinguishes \"The cat sat on the mat\" from \"The sat cat on the mat\".")
IO.puts("===========================================================================\n")
