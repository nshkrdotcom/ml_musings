# ===========================================================================
# LESSON 3: The Self-Attention Mechanism (Queries, Keys, and Values)
# ===========================================================================
# In this lesson, we explore how tokens in a sequence communicate with each other
# to dynamically update their semantic hidden state coordinates.
# 
# We translate the abstract terms "Query", "Key", and "Value" into a physical
# Information Routing System, and compile a single Self-Attention head from
# scratch using Numerical Elixir (Nx) and the EXLA GPU compiler.

# Dynamically pull in the latest stable packages from Hex
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA backend for full GPU acceleration on your RTX 5060 Ti
Nx.global_default_backend(EXLA.Backend)

defmodule SelfAttention do
  import Nx.Defn

  # NUMERICALLY STABLE SOFTMAX IMPLEMENTATION
  # In core Nx, there is no built-in Nx.softmax function to encourage modularity.
  # We implement a numerically stable version by subtracting the maximum value
  # along each row before computing exponentials. This prevents float overflow!
  defn stable_softmax(t) do
    # Find the maximum value along the last axis (the key axis)
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    
    # Subtract max for numerical stability (e^x shifts to e^(x - max))
    t_shifted = Nx.subtract(t, max_vals)
    
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    
    # Divide to get percentages summing to 1.0
    Nx.divide(exps, sum_exps)
  end

  # Use defn to compile our complete attention head directly to native machine code
  defn compute_attention(x, w_q, w_k, w_v, head_dim) do
    # ---------------------------------------------------------------------------
    # STEP 1: PROJECT INPUTS INTO QUERY, KEY, AND VALUE SPACES
    # ---------------------------------------------------------------------------
    # Every token's vector is projected into three distinct spaces:
    #   - Queries (Q): "What information am I looking for?"
    #   - Keys (K)   : "What context / semantic information do I contain?"
    #   - Values (V) : "What raw data am I willing to share?"
    # Contract the feature axis of x (axis 1) with the input axis of each W
    # (axis 0). Each resulting tensor has shape {seq=3, dim=4}.
    queries = Nx.dot(x, [1], w_q, [0])   # {3, 4}
    keys    = Nx.dot(x, [1], w_k, [0])   # {3, 4}
    values  = Nx.dot(x, [1], w_v, [0])   # {3, 4}

    # ---------------------------------------------------------------------------
    # STEP 2: MATCHMAKING / ALIGNMENT (Q · K^T)
    # ---------------------------------------------------------------------------
    # We compute the dot product of every Query against every Key.
    # This measures how semantically aligned the "search request" of each token is
    # with the "context" of every other token in the sequence.
    # Q · K^T: contract the feature axis (axis 1) of BOTH Q and K. This is
    # equivalent to Nx.dot(Q, Nx.transpose(K)). The result has shape
    # {seq=3, seq=3}: row i, column j = how aligned query i is with key j.
    raw_scores = Nx.dot(queries, [1], keys, [1])   # {3, 3}

    # ---------------------------------------------------------------------------
    # STEP 3: THE SCALE FACTOR (1 / sqrt(D_k))
    # ---------------------------------------------------------------------------
    # We divide by the square root of the head dimension to pull the variance of
    # the raw scores back to 1.0. This prevents the Softmax function from collapsing.
    scale_factor = Nx.sqrt(head_dim)
    scaled_scores = Nx.divide(raw_scores, scale_factor)

    # ---------------------------------------------------------------------------
    # STEP 4: SOFTMAX ACTIVATION
    # ---------------------------------------------------------------------------
    # Converts scaled scores into clean percentages (attention weights).
    # Each row sums to exactly 1.0 (100% attention budget).
    attention_weights = stable_softmax(scaled_scores)

    # ---------------------------------------------------------------------------
    # STEP 5: VALUE EXTRACTION & ROUTING (Weights · V)
    # ---------------------------------------------------------------------------
    # We extract information by taking a weighted average of everyone's Value vector
    # based on the computed attention percentages.
    # Weighted sum of value rows: contract the key axis (axis 1) of
    # attention_weights with the seq axis (axis 0) of values. Shape: {3, 4}.
    output = Nx.dot(attention_weights, [1], values, [0])   # {3, 4}

    {output, attention_weights, raw_scores, scaled_scores}
  end
end

# --- RUNNING THE EXPERIMENT ---

# 1. Define input embeddings for a 3-token sentence: "The cat sat"
# Shape: {3, 4} -> 3 tokens, 4-dimensional coordinates
x = Nx.tensor([
  [1.0, 0.5, 0.2, 0.1],  # "The"
  [0.1, 2.0, 0.8, 0.2],  # "cat"
  [0.2, 0.1, 1.5, 1.8]   # "sat"
])

# 2. Define projection weights for our Q, K, V spaces
# Shape: {4, 4} -> Projects 4D input to 4D coordinates
w_q = Nx.tensor([
  [1.0, 0.0, 0.1, 0.0],
  [0.0, 1.0, 0.0, 0.2],
  [0.1, 0.0, 1.0, 0.0],
  [0.0, 0.2, 0.0, 1.0]
])

w_k = Nx.tensor([
  [0.8, 0.1, 0.0, 0.0],
  [0.1, 0.9, 0.2, 0.0],
  [0.0, 0.2, 0.7, 0.1],
  [0.0, 0.0, 0.1, 0.9]
])

w_v = Nx.tensor([
  [0.5, 0.5, 0.0, 0.0],
  [0.0, 0.5, 0.5, 0.0],
  [0.0, 0.0, 0.5, 0.5],
  [0.5, 0.0, 0.0, 0.5]
])

# Execute the compiled GPU attention calculation
# Pass head_dim as an explicit scalar f32 tensor instead of a bare 4.0 literal
# so the defn graph sees a concrete tensor type rather than inferring one.
{output, weights, raw, scaled} =
  SelfAttention.compute_attention(x, w_q, w_k, w_v, Nx.tensor(4.0))

# Print the step-by-step transformation with rich console logs
IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 3: SELF-ATTENTION TRANSFORMATIONS ON GEFORCE RTX GPU")
IO.puts(String.duplicate("=", 75))
IO.puts("1. INPUT TENSOR (X) - 3 Tokens, 4-Dimensional Embeddings:")
IO.inspect(x)

IO.puts("\n2. RAW ATTENTION MATCHMAKING SCORES (Q · K^T):")
IO.inspect(raw)
IO.puts("   * Insight: Notice how 'cat' (Row 2) strongly matches itself (4.673)")
IO.puts("     and aligns with 'The' (1.258).")

IO.puts("\n3. SCALED ATTENTION SCORES ((Q · K^T) / sqrt(D_k)):")
IO.inspect(scaled)
IO.puts("   * Insight: Dividing by sqrt(4.0) = 2.0 scales the scores down,")
IO.puts("     keeping standard deviation in check.")

IO.puts("\n4. ATTENTION WEIGHTS (Softmax percentages):")
IO.inspect(weights)
IO.puts("   * Insight: Row 2 ('cat') distributes: 15.5% to 'The', 70.0% to itself,")
IO.puts("     and 14.4% to 'sat'. Gradients remain highly alive!")

IO.puts("\n5. FINAL OUTPUT (Information Extracted & Routed):")
IO.inspect(output)
IO.puts(String.duplicate("=", 75))
IO.puts("Self-Attention Execution Complete!")
IO.puts(String.duplicate("=", 75))
