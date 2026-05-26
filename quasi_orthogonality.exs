# 1. Dynamically install the Elixir ML packages - Updated to latest Hex versions
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA as the global default backend for fast tensor computation
Nx.global_default_backend(EXLA.Backend)

defmodule QuasiOrthogonality do
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

  # Draw a horizontal ASCII dispersion bar representing the 2-standard-deviation range (~95.4% of pairs)
  # This provides a real-time visual representation of how the dot product distribution concentrates around 0.
  def draw_dispersion_bar(std_val) do
    # Represents the interval [-1.0, 1.0] using a 41-character grid.
    # Index 20 is the center (0.0). Scale is 20 characters per unit of dot product.
    half_width = Float.round(2.0 * std_val * 20.0) |> trunc()
    left = max(0, 20 - half_width)
    right = min(40, 20 + half_width)

    bar =
      Enum.map(0..40, fn
        20 -> "|"
        idx when idx >= left and idx <= right -> "█"
        _ -> "░"
      end)
      |> Enum.join("")

    "[-1.0] #{bar} [+1.0]"
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

    # Extract values back to Elixir native float types
    mean_val = Nx.mean(dot_products) |> Nx.to_number()
    std_val = Nx.standard_deviation(dot_products) |> Nx.to_number()
    pct_not_orthogonal = calculate_not_orthogonal(dot_products, epsilon) |> Nx.to_number()

    theoretical_std = 1.0 / :math.sqrt(dim)
    dispersion_bar = draw_dispersion_bar(std_val)

    # Format and print the outputs cleanly with educational comments
    IO.puts("Dimension (D) = #{String.pad_leading("#{dim}", 5)}")
    IO.puts("  Mean dot product:        #{:erlang.float_to_binary(mean_val, [decimals: 6])}  <-- Centered close to 0 (unbiased directions)")
    IO.puts("  Std Dev (Expected 1/Vd): #{:erlang.float_to_binary(std_val, [decimals: 6])} (Theoretical: #{:erlang.float_to_binary(theoretical_std, [decimals: 6])})")
    IO.puts("  Pairs with |u.v| > #{epsilon}: #{:erlang.float_to_binary(pct_not_orthogonal, [decimals: 2])}%  <-- Fraction of vectors that are NOT quasi-orthogonal")
    IO.puts("  95% Dispersion Bar:      #{dispersion_bar}")
    
    # Render mini-analysis for this step
    cond do
      dim <= 3 ->
        IO.puts("  [Insight]: Almost all vectors are highly correlated. The distribution is widely dispersed.")
      dim <= 64 ->
        IO.puts("  [Insight]: Dispersion is shrinking. A clear boundary of orthogonal vectors is forming.")
      dim <= 512 ->
        IO.puts("  [Insight]: Variance is decaying rapidly. The bulk of the pairs are clustering close to 0.")
      true ->
        IO.puts("  [Insight]: Phenomenon achieved! Vectors are geometrically isolated (almost 100% quasi-orthogonal).")
    end
    IO.puts(String.duplicate("-", 75))
  end
end

# --- Execution Entrypoint & Interactive Tutorial ---
IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 1: THE GEOMETRY OF HIGH-DIMENSIONAL SPACES (QUASI-ORTHOGONALITY)")
IO.puts(String.duplicate("=", 75))
IO.puts("""
This script simulates pairs of random unit vectors in various dimensional spaces
and calculates their dot products (cosine similarity). 

In low dimensions, random vectors overlap heavily. But in high dimensions (D >= 4096),
an unexpected phenomenon occurs: two randomly sampled vectors are almost certainly
perpendicular (quasi-orthogonal). This is the "Blessing of Dimensionality".

We represent the distribution below with a '95% Dispersion Bar' (covering 2 Standard Deviations):
  - [░] represents empty space.
  - [█] represents the high density region where 95% of random dot products lie.
  - [|] represents the center (perfect orthogonality / 0.0 dot product).
""")
IO.puts(String.duplicate("=", 75))

dimensions = [3, 64, 512, 4096, 8192]

Enum.each(dimensions, fn d ->
  QuasiOrthogonality.run(d, 10000, 0.05)
end)

IO.puts("""
=============================== KEY TAKEAWAYS ===============================
1. THE SHAPE OF THE HYPERSPHERE:
   As dimension D increases, the volume of a hypersphere concentrates almost entirely
   along its equator with respect to any arbitrary vector. Consequently, any two random
   vectors are nearly 100% guaranteed to be perpendicular to each other.

2. MATH VERIFICATION (VARIANCE DECAY):
   Notice how the empirical standard deviation scales precisely with 1/sqrt(D). 
   At D = 8192, the theoretical standard deviation is 1/sqrt(8192) ≈ 0.011, meaning 
   95% of all random vector pairs have a dot product of less than ±0.022!

3. PRACTICAL IMPLICATION FOR AI AGENT RUNTIMES & LLMs:
   - Superposition: LLMs pack thousands of different semantic concepts into the same
     activation layer because their representation vectors reside in high-dimensional
     spaces (e.g. D = 4096 or 8192). Quasi-orthogonality ensures these concepts do not
     interfere with each other.
   - Linear Probing & Routing: Linear probes and routers work exceptionally well because
     high-dimensional spaces are easily separable by linear hyperplanes. However, this
     makes overfitting extremely easy (even random noise is linearly separable in high-D).
=============================================================================
""")
