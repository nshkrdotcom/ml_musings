# ===========================================================================
# MINI-LESSON 3: What is a Dot Product? (Geometric Projection & Similarity)
# ===========================================================================
# In machine learning, routing models and search indexes measure how similar
# two thoughts or signals are. The mathematical tool we use for this is the 
# Dot Product (also known as Cosine Similarity when normalized).

# Dynamically pull in Nx dependency - Updated to latest Hex version
Mix.install([{:nx, "~> 0.12.0"}])

defmodule Geometry do
  @moduledoc """
  Toy 1-D-only cosine-similarity helpers.

  PEDAGOGICAL SCOPE: these helpers assume `u`, `v`, and `vector` are RANK-1
  tensors of shape `{D}`. We deliberately do NOT generalize to batched
  inputs (e.g. `{N, D}`) so the lesson keeps focus on the geometric
  intuition of "two arrows in a 2-D plane". A learner who lifts these
  helpers into a batched model must pass `axes: [-1]` (or equivalent) to
  `Nx.sum/2` and `Nx.pow/2` reductions; otherwise the math will silently
  reduce across the batch axis and produce wrong cosine similarities.
  """

  import Nx.Defn

  # 1. NORMALIZATION (1-D vector → unit 1-D vector)
  # If vectors have different lengths, their dot products will scale with length.
  # To focus purely on DIRECTION (semantic meaning), we project the vectors onto
  # a "unit circle" or "unit hypersphere" where every vector has length exactly 1.0.
  defn normalize(vector) do
    # Compute L2 Norm: sqrt(x^2 + y^2 + ...). No axis arg → sums every element.
    # This is intentional for the 1-D contract documented in @moduledoc.
    length = Nx.sum(Nx.pow(vector, 2)) |> Nx.sqrt()
    # Clamp the norm to a minimum of 1.0e-10 before dividing. This avoids
    # division by zero for the zero vector. (The naive `Nx.add(length, 1.0e-10)`
    # alternative leaves a near-infinite output when `length` is 0, instead of
    # the safe near-zero output we get by clamping the denominator.)
    safe_length = Nx.max(length, 1.0e-10)
    Nx.divide(vector, safe_length)
  end

  # 2. COSINE SIMILARITY (DOT PRODUCT, 1-D contract)
  # Multiplying corresponding coordinates and adding them: u·v = u_x*v_x + u_y*v_y
  defn cosine_similarity(u, v) do
    u_norm = normalize(u)
    v_norm = normalize(v)

    # Sum-of-elementwise-product on rank-1 inputs = dot product. Again, no
    # axis arg because the 1-D contract is enforced by convention here.
    Nx.sum(Nx.multiply(u_norm, v_norm))
  end
end

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("STEP 4: THE GEOMETRY OF THE DOT PRODUCT")
IO.puts(String.duplicate("=", 75))
IO.puts("""
We are defining three distinct 2D vectors in a coordinate space:
  - Vector A (pointing Straight Up):    [0.0, 1.0]
  - Vector B (pointing Straight Right): [1.0, 0.0]
  - Vector C (pointing Diagonal Up-Rt): [1.0, 1.0]
""")

pointing_up        = Nx.tensor([0.0, 1.0])
pointing_right     = Nx.tensor([1.0, 0.0])
pointing_up_right  = Nx.tensor([1.0, 1.0])

# Compute and extract similarities as native floats
sim1 = Geometry.cosine_similarity(pointing_up, pointing_right) |> Nx.to_number()
sim2 = Geometry.cosine_similarity(pointing_up, pointing_up_right) |> Nx.to_number()
sim3 = Geometry.cosine_similarity(pointing_up, pointing_up) |> Nx.to_number()

IO.puts("SIMILARITY CALCULATIONS:")
IO.puts(String.duplicate("-", 75))
IO.puts("1. UP vs. RIGHT (90° Angle):")
IO.puts("   Similarity: #{sim1}")
IO.puts("   Result    : Near-zero (floating-point; exactly orthogonal analytically)")
IO.puts("")
IO.puts("2. UP vs. UP-RIGHT (45° Angle):")
IO.puts("   Similarity: #{sim2}")
IO.puts("   Result    : POSITIVE (~0.707 - Highly correlated direction)")
IO.puts("")
IO.puts("3. UP vs. UP (0° Angle):")
IO.puts("   Similarity: #{sim3}")
IO.puts("   Result    : EXACTLY 1.0 (Identical direction / Complete correlation)")
IO.puts(String.duplicate("-", 75))

IO.puts("""
GEOMETRIC INTUITION:

Think of the dot product as a "Shadow Caster":
  - If two vectors are aligned (0° angle), they cast a full shadow on each other
    (Similarity = 1.0).
  - If they are partially aligned (45° angle), they cast a partial shadow 
    (Similarity = 0.707).
  - If they are completely perpendicular (90° angle), they cast NO shadow on
    each other (Similarity = 0.0). They are totally "Orthogonal".

Now that you understand that the dot product measures perpendicularity, you are
ready to understand why "Quasi-Orthogonality" in 8192 dimensions is so powerful.

Proceed to run the main experiment 'quasi_orthogonality.exs' to see this math in high-D!
""")
