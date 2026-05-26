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

    # 2. Extract only the top 'r' components (rank restriction / truncation).
    # NOTE: in Nx 0.12 ranges in Access slicing (`tensor[[range1, range2]]`)
    # are INCLUSIVE on both ends, so `0..rank-1` selects exactly `rank`
    # indices. (Verified against deps/nx Access docs and a smoke test.)
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
  # SHAPES ACTUALLY USED IN THIS FILE (kept consistent with the call site
  # below so the docstring does not lie about what flows through `defn`):
  #
  #   x       : {N, D}    rows are token activations to be adapted
  #   w_0     : {D, D}    frozen pre-trained weight matrix
  #   lora_a  : {r, D}    "down-projection". Plays the role of A in
  #                       Hu et al. 2021 (A is {r, D} there too).
  #   lora_b  : {r, D}    "up-projection ALREADY TRANSPOSED". The Hu et al.
  #                       paper writes B with shape {D, r}; this file passes
  #                       in B^T directly so a single `Nx.dot(..., [1], ..., [0])`
  #                       contraction matches without an extra transpose step.
  #
  # Operationally that means we compute:
  #
  #     Y = X * W_0^T + (alpha / r) * (X * A^T) * B
  #
  # where A^T comes from `Nx.dot(x, [1], lora_a, [1])` and B (= our `lora_b`)
  # gets contracted via `Nx.dot(compressed, [1], lora_b, [0])`. The end-to-end
  # output shape stays {N, D}.
  defn forward(x, w_0, lora_a, lora_b, alpha, rank) do
    # 1. Frozen path: X * W_0^T
    #    Contract axis 1 of x (feature) with axis 1 of W_0 (input).
    #    x: {N, D} -> output_0: {N, D}
    output_0 = Nx.dot(x, [1], w_0, [1])

    # 2. Trainable low-rank path: X * A^T then * B
    #    lora_a is {r, D}; contract axis 1 of x with axis 1 of lora_a.
    #    Result `compressed` has shape {N, r}.
    compressed = Nx.dot(x, [1], lora_a, [1])

    #    lora_b is {r, D} (already-transposed B from the docstring above).
    #    Contract axis 1 of `compressed` (size r) with axis 0 of lora_b (size r).
    #    Result `delta` has shape {N, D}.
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
# lora_a compresses from D=4 to r=1; shape is {r, D} = {1, 4}.
lora_a = Nx.tensor([[0.5, 0.5, 0.5, 0.5]])
# lora_b is the {r, D} up-projection passed in already-transposed (see the
# SHAPES docstring on LoRALayer.forward/6). Its shape here is {1, 4}.
lora_b = Nx.tensor([[1.0, 2.0, 1.0, 0.5]])

final_out = LoRALayer.forward(x, w_0, lora_a, lora_b, 1.0, 1.0)

IO.puts("\n4. INPUT ACTIVATION TENSOR (X):")
IO.inspect(x)

IO.puts("\n5. FINAL LORA OUTPUT (Frozen W_0 Path + Scaled LoRA Low-Rank Delta):")
IO.inspect(final_out)
IO.puts(String.duplicate("=", 75))
IO.puts("SVD and LoRA Execution Complete!")
IO.puts(String.duplicate("=", 75))
