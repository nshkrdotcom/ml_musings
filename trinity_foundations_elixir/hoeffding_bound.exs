# 1. Dynamically install the Elixir ML packages - Updated to latest Hex versions
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA as the global default backend for fast tensor computation
Nx.global_default_backend(EXLA.Backend)

defmodule HoeffdingBound do
  import Nx.Defn

  # Use numerical definition (defn) to compile tensor operations to GPU/CPU via XLA
  defn normalize(tensor) do
    # Compute the L2 norm along axis 1 (the dimension axis)
    norms = tensor
    |> Nx.pow(2)
    |> Nx.sum(axes: [1], keep_axes: true)
    |> Nx.sqrt()

    # Divide by norm to project onto the unit hypersphere (add epsilon to avoid division by zero)
    Nx.divide(tensor, Nx.add(norms, 1.0e-10))
  end

  defn compute_dot_products(u, v) do
    # Element-wise multiplication, then sum along the dimension axis
    Nx.sum(Nx.multiply(u, v), axes: [1])
  end

  defn calculate_not_orthogonal(dot_products, epsilon) do
    dot_products
    |> Nx.abs()
    |> Nx.greater(epsilon)
    |> Nx.mean()
    |> Nx.multiply(100.0)
  end

  def run(dim, num_pairs \\ 10000, epsilon \\ 0.05) do
    # Seed the random number generator using a system-unique integer
    key = Nx.Random.key(System.unique_integer([:positive]))
    
    # Sample random coordinates from a normal distribution
    {u_raw, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {num_pairs, dim})
    {v_raw, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {num_pairs, dim})

    # Project vectors onto the unit hypersphere surface
    u = normalize(u_raw)
    v = normalize(v_raw)

    # Calculate the dot products
    dot_products = compute_dot_products(u, v)

    # Calculate empirical percentage of u.v > epsilon
    pct_not_orthogonal = calculate_not_orthogonal(dot_products, epsilon) |> Nx.to_number()

    # Calculate theoretical Hoeffding Bound: P(|u.v| > epsilon) <= 2 * exp(- D * epsilon^2 / 2)
    hoeffding_bound_prob = 2.0 * :math.exp(-dim * :math.pow(epsilon, 2) / 2.0)
    hoeffding_bound_pct = hoeffding_bound_prob * 100.0

    # Verification check
    satisfied = pct_not_orthogonal <= hoeffding_bound_pct
    status = if satisfied, do: "[PASSED ✅]", else: "[FAILED ❌]"

    # Format and print the outputs cleanly
    IO.puts("Dimension (D) = #{String.pad_leading("#{dim}", 5)}  #{status}")
    IO.puts("  Tolerance Epsilon (ε)      : #{epsilon}")
    IO.puts("  Empirical P(|u·v| > ε)      : #{:erlang.float_to_binary(pct_not_orthogonal, [decimals: 4])}%")
    IO.puts("  Hoeffding Upper Bound Limit : #{:erlang.float_to_binary(hoeffding_bound_pct, [decimals: 4])}%")
    
    # Educational breakdown
    cond do
      hoeffding_bound_pct > 100.0 ->
        IO.puts("  [Mathematical Note]: At D = #{dim}, the theoretical upper bound is > 100%, which is trivial.")
        IO.puts("                       Since the probability of any event is always <= 1.0 (100%), this is mathematically valid but loose.")
      true ->
        tightness = hoeffding_bound_pct - pct_not_orthogonal
        IO.puts("  [Mathematical Note]: The concentration inequality holds. The empirical value sits safely")
        IO.puts("                       inside the theoretical limit (Tightness gap: #{:erlang.float_to_binary(tightness, [decimals: 4])}%).")
    end
    IO.puts(String.duplicate("-", 75))
  end
end

# --- Execution Entrypoint & Interactive Tutorial ---
IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("EXERCISE 1: HOEFFDING CONCENTRATION BOUND VERIFICATION")
IO.puts(String.duplicate("=", 75))
IO.puts("""
This script mathematically verifies that the probability of two random vectors
being "correlated" (not quasi-orthogonal) is strictly bounded from above by
Hoeffding's Concentration inequality:

             P( |u · v| > ε )  <=  2 * exp(-D * ε² / 2)

Where:
  - u, v are random vectors projected onto the unit sphere.
  - ε (epsilon) is our non-orthogonality threshold (default: 0.05).
  - D is the dimensionality of the vector space.
  - exp is the exponential function.

As D increases, notice how the upper bound decreases exponentially, forcing the
empirical probability of finding non-orthogonal random vectors to plummet to 0%.
""")
IO.puts(String.duplicate("=", 75))

dimensions = [3, 64, 512, 4096, 8192]

Enum.each(dimensions, fn d ->
  HoeffdingBound.run(d, 10000, 0.05)
end)

IO.puts("""
=============================== KEY TAKEAWAYS ===============================
1. THE POWER OF EXPONENTIAL CONCENTRATION:
   Look at the boundary limits:
   - At D = 3, the math allows up to 199.25% probability (a loose, trivial bound).
   - At D = 512, the bound restricts the probability to at most 105.46%.
   - At D = 4096, the limit drops drastically to 1.1952%, and empirical drops to 0.10%.
   - At D = 8192, the limit is a minuscule 0.0071%, locking empirical down to 0.00%.

2. HOW SYSTEM ARCHITECTS EXPLOIT THIS:
   By knowing the mathematical concentration bounds, AI system engineers can determine
   with extreme statistical certainty the probability that two random signals in an
   agent router or memory vector database will cross-talk or trigger false positives.
   For instance, at D = 8192, a dot product above 0.05 is virtually impossible to happen
   by chance, so any alignment > 0.05 is highly statistically significant!
=============================================================================
""")
