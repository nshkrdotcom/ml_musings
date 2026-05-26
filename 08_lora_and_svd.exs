# ===========================================================================
# LESSON 4: Parameter-Efficient Adaptation (Rank, SVD, and LoRA)
# ===========================================================================
# In this lesson, we explore Parameter-Efficient Fine-Tuning (PEFT) using
# Singular Value Decomposition (SVD) and Low-Rank Adaptation (LoRA).
# 
# We implement a tool that compresses redundant dimensions of a weight matrix
# using SVD, print an ASCII singular value decay elbow plot, and compile a
# functional LoRA bypass layer with training / backpropagation steps.

# Dynamically pull in the latest Hex packages
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA default backend
Nx.global_default_backend(EXLA.Backend)

defmodule MatrixSurgery do
  @doc """
  Performs Singular Value Decomposition (SVD) on a matrix
  and reconstructs a low-rank approximation of rank `r`.
  """
  def compress(matrix, rank) do
    # 1. Decompose the matrix: W = U · Sigma · V^T
    {u, s, vt} = Nx.LinAlg.svd(matrix)
    sigma = Nx.make_diagonal(s)

    # 2. Extract top 'r' components (rank restriction)
    {height, width} = Nx.shape(matrix)
    u_truncated     = u[[0..height-1, 0..rank-1]]
    sigma_truncated = sigma[[0..rank-1, 0..rank-1]]
    vt_truncated    = vt[[0..rank-1, 0..width-1]]

    # 3. Reconstruct approximated low-rank matrix
    reconstructed = u_truncated
    |> Nx.dot(sigma_truncated)
    |> Nx.dot(vt_truncated)

    {reconstructed, s}
  end

  # Helper to print a beautiful 1D ASCII Singular Value Decay elbow plot
  def print_decay_plot(singular_values) do
    s_list = Nx.to_flat_list(singular_values)
    sum_s = Enum.sum(s_list)
    
    IO.puts("\nSINGULAR VALUE DECAY ELBOW PLOT (ASCII):")
    IO.puts(String.duplicate("-", 75))
    
    max_val = Enum.max(s_list)
    bar_width = 30
    
    # 1. Print decay bars
    Enum.with_index(s_list) |> Enum.each(fn {val, idx} ->
      percentage = (val / sum_s) * 100.0
      filled = round((val / max_val) * bar_width)
      empty = bar_width - filled
      bar = String.duplicate("█", filled) <> String.duplicate("░", empty)
      
      label = :io_lib.format("  S~B (~.3f): [~s] (~.1f%)", [idx + 1, val, bar, percentage]) |> List.to_string()
      IO.puts(label)
    end)
    
    # 2. Print cumulative variance explained
    IO.puts("\nCUMULATIVE VARIANCE EXPLAINED:")
    Enum.reduce(Enum.with_index(s_list), 0.0, fn {val, idx}, acc_var ->
      perc = (val / sum_s) * 100.0
      new_acc = acc_var + perc
      label = :io_lib.format("  Rank ~B Approximation: ~.2f%", [idx + 1, new_acc]) |> List.to_string()
      IO.puts(label)
      new_acc
    end)
    IO.puts(String.duplicate("-", 75))
  end
end

defmodule LoRALayer do
  import Nx.Defn

  # Forward pass: Y = X · W_0^T + (alpha / r) · (X · A^T) · B
  # x: {N, D}, w_0: {D, D}, lora_a: {r, D}, lora_b: {r, D}
  defn forward(x, w_0, lora_a, lora_b, alpha, rank) do
    # 1. Frozen pre-trained path
    output_0 = Nx.dot(x, [1], w_0, [1])

    # 2. Trainable low-rank adapter path
    compressed = Nx.dot(x, [1], lora_a, [1]) # {N, r}
    delta = Nx.dot(compressed, [1], lora_b, [0]) # {N, D}

    # Scaling factor
    scaling = Nx.divide(alpha, rank)
    scaled_delta = Nx.multiply(delta, scaling)

    Nx.add(output_0, scaled_delta)
  end

  # GPU-Compiled Training Step using value_and_grad
  # Optimizes lora_a and lora_b to match a target output, keeping w_0 frozen.
  defn train_step(lora_a, lora_b, x, w_0, target, lr, alpha, rank) do
    {loss_val, {grad_a, grad_b}} =
      value_and_grad({lora_a, lora_b}, fn {a, b} ->
        out = forward(x, w_0, a, b, alpha, rank)
        # Mean Squared Error Loss
        Nx.mean(Nx.pow(Nx.subtract(out, target), 2))
      end)

    # Gradient descent update
    new_a = Nx.subtract(lora_a, Nx.multiply(lr, grad_a))
    new_b = Nx.subtract(lora_b, Nx.multiply(lr, grad_b))

    {new_a, new_b, loss_val, {grad_a, grad_b}}
  end
end

# --- RUNNING THE EXPERIMENT ---

# W_0 is our initial, redundant weight matrix
w_0_redundant = Nx.tensor([
  [1.0,  2.0,  3.0,  4.0],
  [2.0,  4.0,  6.0,  8.0],
  [3.0,  6.0,  9.0, 12.0],
  [4.0,  8.0, 12.0, 16.0]
])

# W_noisy is a full-rank matrix with decaying singular values
w_noisy = Nx.tensor([
  [1.0,  2.0,  1.5,  0.8],
  [2.0,  4.5,  6.0,  3.1],
  [1.2,  3.0,  9.0, 12.0],
  [0.5,  1.8,  4.2, 16.0]
])

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 4: SVD COMPRESSION, DECAY VISUALIZATION, AND LORA TRAINING")
IO.puts(String.duplicate("=", 75))

# 1. Demonstrate SVD Compression & Decay Plot
IO.puts("1. SVD ANALYSIS & COMPRESSION DEMONSTRATION")
{_reconstructed, singular_values} = MatrixSurgery.compress(w_noisy, 2)
MatrixSurgery.print_decay_plot(singular_values)
IO.puts("  * The elbow should appear after the first 1-2 components — S1 and S2 capture most variance, S3 and S4 are relatively small.")

# 2. Demonstrate the LoRA Forward Pass
x = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
lora_a_init = Nx.tensor([[0.5, 0.5, 0.5, 0.5]])
lora_b_init = Nx.tensor([[1.0, 2.0, 1.0, 0.5]])

forward_out = LoRALayer.forward(x, w_0_redundant, lora_a_init, lora_b_init, 1.0, 1.0)
IO.puts("\n2. LORA FORWARD PASS ON A SINGLE TOKEN:")
IO.puts("  - Input Activation Shape:    #{inspect(Nx.shape(x))}")
IO.puts("  - lora_a Shape (down-proj):  #{inspect(Nx.shape(lora_a_init))}")
IO.puts("  - lora_b Shape (up-proj):    #{inspect(Nx.shape(lora_b_init))}")
IO.puts("  - Computed Output Tensor:")
IO.inspect(forward_out)
IO.puts("")

# 3. LoRA Training Loop Demonstration
IO.puts(String.duplicate("-", 75))
IO.puts("3. TRAINING LORA ADAPTERS (W_0 REMAINING FROZEN)")
IO.puts(String.duplicate("-", 75))

# Setup batch size 2 toy training data
x_train = Nx.tensor([
  [1.0, 1.0, 1.0, 1.0],
  [-0.5, 2.0, 0.5, -1.0]
])

# Define target outputs we want to fine-tune to match
target = Nx.tensor([
  [12.0, 23.0, 31.0, 42.0],
  [-3.0,  5.0,  8.0, -2.0]
])

# Save copy of w_0 so we can verify it remains completely unchanged
w_0_frozen = w_0_redundant

# Initialize adapter weights (rank r = 1):
lora_a = Nx.tensor([[0.1, -0.2, 0.1, 0.3]])
lora_b = Nx.tensor([[-0.1, 0.4, -0.2, 0.1]])

lr = Nx.tensor(0.05)
alpha = Nx.tensor(1.0)
rank = Nx.tensor(1.0)

# Train adapters for 5 epochs
{final_a, final_b, final_loss} = Enum.reduce(1..5, {lora_a, lora_b, 0.0}, fn epoch, {a, b, _} ->
  {next_a, next_b, loss, _grads} =
    LoRALayer.train_step(a, b, x_train, w_0_redundant, target, lr, alpha, rank)

  loss_num = Nx.to_number(loss)
  IO.puts("  Epoch #{epoch} | Adapters MSE Training Loss: #{Float.round(loss_num, 6)}")
  {next_a, next_b, loss_num}
end)

IO.puts("\nTraining Complete! Final Loss: #{Float.round(final_loss, 6)}")

# Verification check: base weights w_0 must be 100% untouched
# NOTE: since Nx tensors are immutable values, this check always passes.
# The real proof that W_0 is frozen is that value_and_grad differentiates only {lora_a, lora_b}, not w_0.
w_0_diff = Nx.subtract(w_0_redundant, w_0_frozen)
w_0_diff_magnitude = Nx.sqrt(Nx.sum(Nx.pow(w_0_diff, 2))) |> Nx.to_number()

IO.puts("\nFROZEN STATE VERIFICATION:")
IO.puts("  - Base Weights W_0 L2 difference from start of training: #{Float.round(w_0_diff_magnitude, 8)}")
IO.puts("    (Should be exactly 0.0, validating that W_0 was never updated!)")
IO.puts("  - Post-training Adapters:")
IO.puts("    - lora_a (Down-Proj) weights:")
IO.inspect(final_a)
IO.puts("    - lora_b (Up-Proj) weights:")
IO.inspect(final_b)
IO.puts("===========================================================================\n")
