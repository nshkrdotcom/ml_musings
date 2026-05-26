# ===========================================================================
# LESSON 4: Parameter-Efficient Adaptation (Rank, SVD, and LoRA)
# ===========================================================================
# In this lesson, we explore Parameter-Efficient Fine-Tuning (PEFT) using
# Singular Value Decomposition (SVD) and Low-Rank Adaptation (LoRA).
# 
# We implement a tool that compresses redundant dimensions of a weight matrix
# using SVD, and compile a functional LoRA bypass layer from scratch using
# Numerical Elixir (Nx) and the EXLA GPU compiler.

# Dynamically pull in the latest stable dependencies from Hex
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA backend to compile all computations directly to CUDA GPU kernels
Nx.global_default_backend(EXLA.Backend)

defmodule MatrixSurgery do
  @doc """
  Performs Singular Value Decomposition (SVD) on a matrix
  and reconstructs a low-rank approximation of rank `r`.
  """
  def compress(matrix, rank) do
    # 1. Decompose the matrix: W = U · Sigma · V^T
    #   - u: Left singular vectors (output coordinate directions)
    #   - s: 1D vector of singular values (eigen-energy / importance)
    #   - vt: Right singular vectors (input coordinate directions)
    {u, s, vt} = Nx.LinAlg.svd(matrix)

    # Convert the 1D list of singular values into a diagonal matrix using Nx.make_diagonal
    sigma = Nx.make_diagonal(s)

    # 2. Extract only the top 'r' components (rank restriction / truncation)
    #    In Elixir's Nx, multidimensional slicing requires double brackets: tensor[[range1, range2]]
    {height, width} = Nx.shape(matrix)
    u_truncated     = u[[0..height-1, 0..rank-1]]
    sigma_truncated = sigma[[0..rank-1, 0..rank-1]]
    vt_truncated    = vt[[0..rank-1, 0..width-1]]

    # 3. Reconstruct the approximated low-rank matrix
    reconstructed = u_truncated
    |> Nx.dot(sigma_truncated)
    |> Nx.dot(vt_truncated)

    {reconstructed, s}
  end
end

defmodule LoRALayer do
  import Nx.Defn

  # Use defn to compile our forward pass to native machine code.
  #
  # CONVENTION (Hu et al. 2021):
  #   h = W_0 x + (alpha / r) * B A x       where
  #     W_0 has shape {D, D}              (frozen pre-trained weights)
  #     A   has shape {r, D}              (down-projection)
  #     B   has shape {D, r}              (up-projection)
  #
  # In this implementation we batch over rows (x is shape {N, D}, one row per
  # token), so the equivalent computation is:
  #   Y = X * W_0^T + (alpha / r) * (X * A^T) * B^T
  # which is what `Nx.dot(... [1], ..., [1])` expresses below.
  defn forward(x, w_0, lora_a, lora_b, alpha, rank) do
    # 1. Frozen path: X * W_0^T
    #    Contract axis 1 of x (feature) with axis 1 of W_0 (input).
    #    x: {N, D} -> output_0: {N, D}
    output_0 = Nx.dot(x, [1], w_0, [1])

    # 2. Trainable low-rank path: X * A^T then * B^T
    #    lora_a plays the role of A (shape: {r, D}); contract axis 1 of x with axis 1 of A.
    #    Result: {N, r}.
    compressed = Nx.dot(x, [1], lora_a, [1])

    #    lora_b is the {r, D} matrix used to project compressed back up; contract axis 1
    #    of compressed (dimension r) with axis 0 of lora_b (dimension r). Result: {N, D}.
    #
    #    NOTE on naming: a literal Hu-et-al "B" would have shape {D, r}. Here we keep the
    #    variable name `lora_b` for parity with the paper's vocabulary, but the tensor we
    #    actually pass at the call site is shape {r, D} and represents B^T directly. The
    #    contraction axes below are consistent with that.
    delta = Nx.dot(compressed, [1], lora_b, [0])

    # 3. Scale the low-rank delta by alpha / r (LoRA paper's standard scaling).
    scaling = Nx.divide(alpha, rank)
    scaled_delta = Nx.multiply(delta, scaling)

    # 4. Sum the frozen and the scaled trainable paths.
    Nx.add(output_0, scaled_delta)
  end
end

# --- RUNNING THE EXPERIMENT ---

# Create a highly redundant 4x4 weight matrix W_0
# Notice that Row 2, 3, and 4 are simple scalar multiples of Row 1 (Rank-1 matrix!)
w_0 = Nx.tensor([
  [1.0,  2.0,  3.0,  4.0],
  [2.0,  4.0,  6.0,  8.0],
  [3.0,  6.0,  9.0, 12.0],
  [4.0,  8.0, 12.0, 16.0]
])

# 1. Demonstrate SVD Compression
IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 4: SVD MATRIX SURGERY & LORA BYPASS CALCULATIONS ON GPU")
IO.puts(String.duplicate("=", 75))
IO.puts("1. ORIGINAL REDUNDANT WEIGHT MATRIX W_0 (4x4):")
IO.inspect(w_0)

{reconstructed, singular_values} = MatrixSurgery.compress(w_0, 1)

IO.puts("\n2. SINGULAR VALUES (Eigen-energy / Variance of each coordinate direction):")
IO.inspect(singular_values)
IO.puts("   * Insight: Notice that only the first singular value is non-zero (30.0)!")
IO.puts("     The remaining 3 directions contain exactly 0.0 energy. The matrix is Rank-1.")

IO.puts("\n3. RECONSTRUCTED MATRIX (Truncated to Rank-1):")
IO.inspect(reconstructed)
IO.puts("   * Insight: The reconstruction is 100% identical to the original weight matrix")
IO.puts("     because all other dimensions were completely redundant.")

# 2. Demonstrate the LoRA Forward Pass
# Input vector representing a token activation (Shape: {1, 4})
x = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])

# Define a Rank-1 LoRA Bypass (rank r = 1)
# lora_a compresses from D=4 to r=1 (Shape: {1, 4})
lora_a = Nx.tensor([[0.5, 0.5, 0.5, 0.5]])
# lora_b expands from r=1 back up to D=4. As constructed below it is shape
# {1, 4}, which represents B^T in the standard convention (B itself would be
# shape {4, 1}). The contraction axes inside `forward/6` are matched to this
# convention; see the docstring on LoRALayer.forward/6.
lora_b = Nx.tensor([[1.0, 2.0, 1.0, 0.5]])

final_out = LoRALayer.forward(x, w_0, lora_a, lora_b, 1.0, 1.0)

IO.puts("\n4. INPUT ACTIVATION TENSOR (X):")
IO.inspect(x)

IO.puts("\n5. FINAL LORA OUTPUT (Frozen W_0 Path + Scaled LoRA Low-Rank Delta):")
IO.inspect(final_out)
IO.puts(String.duplicate("=", 75))
IO.puts("SVD and LoRA Execution Complete!")
IO.puts(String.duplicate("=", 75))
