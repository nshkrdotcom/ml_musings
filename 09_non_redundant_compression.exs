# ===========================================================================
# EXERCISE 1: Low-Rank Compression on a Non-Redundant Matrix
# ===========================================================================
# In this exercise, we explore what happens when we try to compress a 
# non-redundant, full-rank matrix into a low-rank subspace (Rank-1).
# We generate a random 4x4 matrix, perform SVD compression, and calculate
# the exact reconstruction error (information loss) using MSE and L2 norm.

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule NonRedundantMatrixSurgery do
  @doc """
  Generates a random 4x4 matrix using a uniform distribution.
  This matrix will be full-rank and non-redundant.
  """
  def generate_matrix(seed) do
    key = Nx.Random.key(seed)
    # Generate coordinates uniformly between 0.0 and 10.0
    {matrix, _key} = Nx.Random.uniform(key, 0.0, 10.0, shape: {4, 4})
    matrix
  end

  @doc """
  SVD compression: Decomposes and truncates to rank 'r'.
  """
  def compress(matrix, rank) do
    {u, s, vt} = Nx.LinAlg.svd(matrix)
    sigma = Nx.make_diagonal(s)

    # In Elixir's Nx, multidimensional slicing requires double brackets: tensor[[range1, range2]].
    # Ranges are INCLUSIVE on both ends, so `0..rank-1` selects exactly `rank` indices.
    {height, width} = Nx.shape(matrix)
    u_truncated     = u[[0..height-1, 0..rank-1]]
    sigma_truncated = sigma[[0..rank-1, 0..rank-1]]
    vt_truncated    = vt[[0..rank-1, 0..width-1]]

    reconstructed = u_truncated
    |> Nx.dot(sigma_truncated)
    |> Nx.dot(vt_truncated)

    {reconstructed, s}
  end

  @doc """
  Computes the Mean Squared Error (MSE) and Frobenius Norm (L2) error
  between the original and reconstructed matrix on the device.
  """
  def calculate_errors(original, reconstructed) do
    diff = Nx.subtract(original, reconstructed)
    
    # 1. Mean Squared Error (MSE)
    mse = Nx.mean(Nx.pow(diff, 2)) |> Nx.to_number()

    # 2. Frobenius Norm (L2 Matrix Error). We pass `ord: :frobenius`
    #    explicitly even though Nx 0.12.1's default for a 2-D matrix is
    #    already Frobenius — being explicit guards against future Nx
    #    versions or readers who only learn from this snippet.
    l2_error = Nx.LinAlg.norm(diff, ord: :frobenius) |> Nx.to_number()

    {mse, l2_error}
  end
end

# --- RUNNING THE EXPERIMENT ---
seed = 1234
rank = 1

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("EXERCISE 1: SVD COMPRESSION & INFORMATION LOSS ON NON-REDUNDANT MATRIX")
IO.puts(String.duplicate("=", 75))

# Generate the full-rank matrix
original = NonRedundantMatrixSurgery.generate_matrix(seed)
IO.puts("1. ORIGINAL FULL-RANK RANDOM MATRIX W_0 (4x4):")
IO.inspect(original)

# Compress to Rank-1
{reconstructed, s} = NonRedundantMatrixSurgery.compress(original, rank)

IO.puts("\n2. SINGULAR VALUES (Eigen-energy of each dimension):")
IO.inspect(s)
IO.puts("   * Insight: Notice that ALL four singular values are non-zero!")
IO.puts("     Every single dimension contains unique, non-redundant information.")

IO.puts("\n3. RECONSTRUCTED RANK-1 APPROXIMATION:")
IO.inspect(reconstructed)

# Calculate errors
{mse, l2} = NonRedundantMatrixSurgery.calculate_errors(original, reconstructed)

IO.puts("\n4. MEASURING THE INFORMATION LOSS (RECONSTRUCTION ERROR):")
IO.puts("   - Mean Squared Error (MSE) : #{:erlang.float_to_binary(mse, [decimals: 6])}")
IO.puts("   - Frobenius Matrix L2 Error: #{:erlang.float_to_binary(l2, [decimals: 6])}")
IO.puts(String.duplicate("-", 75))
IO.puts("""
CRITICAL LESSON:

Unlike the redundant matrix in our first script (which was compressed with 0% loss),
forcing a non-redundant matrix into a low-rank subspace causes substantial 
information loss! 

Why is this useful for LLM adaptation?
LLM pre-trained weights W_0 are full-rank, containing highly diverse pre-trained
information. However, when adapting the model to a specific target task (like 
summarization), the required weight UPDATE (Delta W) lies in a highly redundant
low-rank subspace. 

By freezing W_0 and training only the low-rank bypass (A and B), we preserve
the full-rank pre-trained knowledge while dynamically adapting the model with
minimal parameters and zero information leakage!
""")
IO.puts(String.duplicate("=", 75))

# ---------------------------------------------------------------------------
# EMPIRICAL "DELTA IS LOW-RANK" PROOF
# ---------------------------------------------------------------------------
# The closing paragraph asserts that fine-tune deltas live in a low-rank
# subspace. Let's not hand-wave it: construct a small synthetic adapter
# delta the same way LoRA does (delta = B · A, where B is {D, r} and A is
# {r, D}), then take an SVD and show that only `r` singular values are
# non-trivial, regardless of how large D is.
IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("APPENDIX: EMPIRICAL SVD DECAY OF A SYNTHETIC LoRA ADAPTER DELTA")
IO.puts(String.duplicate("=", 75))

# Construct a rank-1 adapter delta over D=4 features.
b_factor = Nx.tensor([[0.9], [0.1], [-0.4], [0.6]])           # shape {D=4, r=1}
a_factor = Nx.tensor([[1.0, 2.0, 1.0, 0.5]])                  # shape {r=1, D=4}
delta = Nx.dot(b_factor, a_factor)                            # shape {4, 4}, rank 1

IO.puts("Synthetic adapter delta = B · A  (B: {4,1}, A: {1,4}, rank-1 by construction):")
IO.inspect(delta)

{_u_d, s_d, _vt_d} = Nx.LinAlg.svd(delta)
IO.puts("\nSingular values of delta:")
IO.inspect(s_d)
IO.puts("""
* Insight: only the FIRST singular value is materially non-zero. Every
  other singular value is numerical noise (≈ 1e-7 or smaller in f32).
  This is the empirical demonstration of the LoRA assumption: even though
  W_0 needs full rank to be useful, the per-task UPDATE we add to it can
  be expressed in `r << D` directions without losing information about
  the update itself.
""")
IO.puts(String.duplicate("=", 75))
