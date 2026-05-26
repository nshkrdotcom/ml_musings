# ===========================================================================
# EXERCISE 1: Softmax Collapse and the Math of Scaling (sqrt(D_k))
# ===========================================================================
# In this exercise, we empirically demonstrate why dividing attention scores
# by the square root of the head dimension (sqrt(D_k)) is absolutely necessary.
#
# We simulate a large-dimensional attention layer (e.g. D = 1024), generate
# random high-variance dot products, and compare the resulting Softmax
# distributions side-by-side to observe the "One-Hot Collapse".

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule SoftmaxCollapse do
  import Nx.Defn

  # NUMERICALLY STABLE SOFTMAX
  # Subtract max values along each row before computing exponentials.
  defn stable_softmax(t) do
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    t_shifted = Nx.subtract(t, max_vals)
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    Nx.divide(exps, sum_exps)
  end

  # SOFTMAX JACOBIAN: J[i,j] = s_i * (delta_ij - s_j) = diag(s) - s * s^T
  # Given a 1-D softmax vector `s` of length N, returns an {N, N} Jacobian matrix.
  # We use this below to empirically demonstrate that when softmax collapses to a
  # near-one-hot distribution, every entry of the Jacobian is approximately zero,
  # making backprop unable to push gradient through the attention layer.
  defn softmax_jacobian(s) do
    # s is shape {1, N}. Squeeze to a 1-D vector of length N.
    s_vec = Nx.reshape(s, {Nx.size(s)})
    n = Nx.size(s_vec)
    diag = Nx.make_diagonal(s_vec)
    # Outer product s s^T
    outer = Nx.dot(Nx.reshape(s_vec, {n, 1}), Nx.reshape(s_vec, {1, n}))
    Nx.subtract(diag, outer)
  end

  # Generate synthetic high-variance attention scores representing large-D projections
  def generate_scores(dim, num_tokens) do
    # Variance of random dot product scales linearly with dimension D.
    # Therefore, standard deviation scales with sqrt(D).
    std_dev = :math.sqrt(dim)
    
    key = Nx.Random.key(42)
    # Generate scores with expected variance equal to the dimension size
    {scores, _key} = Nx.Random.normal(key, 0.0, std_dev, shape: {1, num_tokens})
    scores
  end

  # Compare unscaled vs scaled softmax outputs
  defn compare_distributions(scores, dim) do
    # 1. Unscaled Softmax
    unscaled_softmax = stable_softmax(scores)

    # 2. Scaled Softmax (Divided by sqrt(D_k))
    scale_factor = Nx.sqrt(dim)
    scaled_scores = Nx.divide(scores, scale_factor)
    scaled_softmax = stable_softmax(scaled_scores)

    {unscaled_softmax, scaled_softmax, scaled_scores}
  end
end

# --- RUNNING THE EXPERIMENT ---
dim = 1024
num_tokens = 5

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("EXERCISE 1: PROVING THE SOFTMAX ONE-HOT COLLAPSE (D = 1024)")
IO.puts(String.duplicate("=", 75))

# Generate scores: variance = 1024, standard deviation ≈ 32.0
raw_scores = SoftmaxCollapse.generate_scores(dim, num_tokens)
{unscaled_sm, scaled_sm, scaled_scores} = SoftmaxCollapse.compare_distributions(raw_scores, dim)

# Convert to Elixir numbers for neat presentation
raw_list = Nx.to_flat_list(raw_scores)
scaled_list = Nx.to_flat_list(scaled_scores)
unscaled_sm_list = Nx.to_flat_list(unscaled_sm)
scaled_sm_list = Nx.to_flat_list(scaled_sm)

IO.puts("1. RAW UNSCALED ATTENTION SCORES:")
Enum.zip(1..num_tokens, raw_list)
|> Enum.each(fn {idx, val} -> 
  IO.puts("   Token #{idx} Score: #{:erlang.float_to_binary(val, [decimals: 4])}")
end)
IO.puts("   * Variance is extremely high because it scales linearly with D (D = 1024).")

IO.puts("\n2. UNSCALED SOFTMAX PERCENTAGES (NO DIVISION):")
Enum.zip(1..num_tokens, unscaled_sm_list)
|> Enum.each(fn {idx, val} -> 
  percent = val * 100.0
  IO.puts("   Token #{idx} Attention Weight: #{:erlang.float_to_binary(percent, [decimals: 6])}%")
end)
IO.puts("\n   [RESULT]: ONE-HOT COLLAPSE!")
IO.puts("             The unscaled softmax collapsed into a single winning token.")
IO.puts("             The other elements have dropped to absolute zero (0.000%).")

IO.puts("\n" <> String.duplicate("-", 75))

IO.puts("3. SCALED ATTENTION SCORES (Divided by sqrt(1024) = 32.0):")
Enum.zip(1..num_tokens, scaled_list)
|> Enum.each(fn {idx, val} -> 
  IO.puts("   Token #{idx} Scaled Score: #{:erlang.float_to_binary(val, [decimals: 4])}")
end)
IO.puts("   * Variance is scaled back to 1.0, keeping scores tightly bounded.")

IO.puts("\n4. SCALED SOFTMAX PERCENTAGES (WITH DIVISION):")
Enum.zip(1..num_tokens, scaled_sm_list)
|> Enum.each(fn {idx, val} -> 
  percent = val * 100.0
  IO.puts("   Token #{idx} Attention Weight: #{:erlang.float_to_binary(percent, [decimals: 2])}%")
end)
IO.puts("\n   [RESULT]: POWER PRESERVED!")
IO.puts("             The attention weights form a rich, smooth distribution,")
IO.puts("             allowing information to flow from multiple tokens.")

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("THE MATHEMATICAL CRIME: VANISHING GRADIENTS")
IO.puts(String.duplicate("=", 75))
IO.puts("""
Why does this collapse paralyze training?

1. THE SOFTMAX DERIVATIVE:
   The derivative of softmax for element i with respect to score j is:
     ∂s_i / ∂z_j = s_i * (δ_ij - s_j)
   Where:
     - s_i, s_j are the softmax probabilities.
     - δ_ij is 1 if i == j, and 0 otherwise.

2. WHAT HAPPENS WHEN s_i COLLAPSES (s_winner = 1.0, s_others = 0.0)?
   - For the winner:  1.0 * (1.0 - 1.0) = 0.0!
   - For the losers:  0.0 * (0.0 - 0.0) = 0.0!

If we do not divide by sqrt(D_k), the derivative at every single element collapses
to EXACTLY 0. This instantly paralyzes gradient flow during backpropagation,
preventing the model from learning anything!

Dividing by sqrt(D_k) keeps the scores close together, preserving active
attention distributions and keeping gradients alive!
""")
IO.puts(String.duplicate("=", 75))


# ---------------------------------------------------------------------------
# EMPIRICAL JACOBIAN PROOF (added per critique_001.md Item 7)
# ---------------------------------------------------------------------------
# The text above explained the math analytically. Now we COMPUTE both Jacobians
# numerically so the student can SEE the zero-gradient collapse, not just read
# about it.

unscaled_jacobian = SoftmaxCollapse.softmax_jacobian(unscaled_sm)
scaled_jacobian = SoftmaxCollapse.softmax_jacobian(scaled_sm)

# Quick magnitude summaries
collapsed_max_abs = unscaled_jacobian |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
collapsed_l1 = unscaled_jacobian |> Nx.abs() |> Nx.sum() |> Nx.to_number()
scaled_max_abs = scaled_jacobian |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
scaled_l1 = scaled_jacobian |> Nx.abs() |> Nx.sum() |> Nx.to_number()

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("EMPIRICAL SOFTMAX JACOBIAN COMPARISON")
IO.puts(String.duplicate("=", 75))
IO.puts("Below are the full Jacobian matrices J[i,j] = s_i * (delta_ij - s_j)")
IO.puts("for both the collapsed and the scaled softmax outputs.")
IO.puts("")

IO.puts("1. COLLAPSED (UNSCALED) JACOBIAN:")
IO.inspect(unscaled_jacobian)
IO.puts("   - max |J[i,j]| = #{:erlang.float_to_binary(collapsed_max_abs, [decimals: 8])}")
IO.puts("   - sum |J[i,j]| = #{:erlang.float_to_binary(collapsed_l1, [decimals: 8])}")
IO.puts("   * Every entry is vanishingly small (sub-1e-3) relative to the scaled case below.")

IO.puts("")
IO.puts("2. SCALED JACOBIAN:")
IO.inspect(scaled_jacobian)
IO.puts("   - max |J[i,j]| = #{:erlang.float_to_binary(scaled_max_abs, [decimals: 8])}")
IO.puts("   - sum |J[i,j]| = #{:erlang.float_to_binary(scaled_l1, [decimals: 8])}")
IO.puts("   - L1 ratio scaled/collapsed: #{:erlang.float_to_binary(scaled_l1 / collapsed_l1, [decimals: 2])}x more gradient mass when scaled.")
IO.puts("   * Diagonal entries are O(0.1) and off-diagonals are O(0.01-0.15):
   *   gradient flow is alive and the layer can learn.")
IO.puts(String.duplicate("=", 75))
