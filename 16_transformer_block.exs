# ===========================================================================
# LESSON 4b: The Full Transformer Block (Pre-Norm, MHA, and Gated MLP)
# ===========================================================================
# This script constructs a complete, modern Transformer block from scratch.
# It brings together the major architectural features analyzed in Foundation D:
#   - LayerNorm and RMSNorm (numerical stabilization)
#   - Multi-Head Causal Attention (parallel relational heads + shape transposition)
#   - Gated MLP with SwiGLU activations (Llama/Mistral style feature computation)
#   - Pre-Norm Layout and Additive Residual Connections
#
# Shape tracking matches the exact shape stories in lessons 0020 and 0023.

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule Normalization do
  import Nx.Defn

  # LayerNorm implementation: (x - mean) / (std + eps) * gamma + beta
  # x: {batch, seq, hidden_dim}
  defn layer_norm(x, gamma, beta, eps \\ 1.0e-5) do
    # Compute mean along the hidden dimension (axis -1)
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    
    # Compute variance: mean((x - mean)^2)
    diff = Nx.subtract(x, mean)
    variance = Nx.mean(Nx.pow(diff, 2), axes: [-1], keep_axes: true)
    
    # Normalize
    x_norm = Nx.divide(diff, Nx.sqrt(Nx.add(variance, eps)))
    
    # Scale and shift
    Nx.add(Nx.multiply(x_norm, gamma), beta)
  end

  # RMSNorm implementation: x / rms(x) * gamma
  # Omits mean subtraction, normalizing only by root-mean-square
  defn rms_norm(x, gamma, eps \\ 1.0e-5) do
    # rms = sqrt(mean(x^2))
    variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    x_norm = Nx.divide(x, Nx.sqrt(Nx.add(variance, eps)))
    
    # Scale only
    Nx.multiply(x_norm, gamma)
  end
end

defmodule TransformerBlock do
  import Nx.Defn

  @num_heads 2
  @head_dim 4

  # Numeric stable softmax
  defn stable_softmax(t) do
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    t_shifted = Nx.subtract(t, max_vals)
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    Nx.divide(exps, sum_exps)
  end

  # Multi-Head Causal Attention
  # x: {batch, seq, hidden_dim}
  defn causal_mha(x, w_q, w_k, w_v, w_o) do
    batch_size = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    # 1. Project to Q, K, V
    # Shapes: W_q, W_k, W_v are {hidden_dim, hidden_dim}
    q_all = Nx.dot(x, [2], w_q, [0]) # {batch, seq, hidden_dim}
    k_all = Nx.dot(x, [2], w_k, [0]) # {batch, seq, hidden_dim}
    v_all = Nx.dot(x, [2], w_v, [0]) # {batch, seq, hidden_dim}

    # 2. Reshape and Transpose to split into heads
    # {batch, seq, hidden_dim} -> {batch, seq, heads, head_dim} -> {batch, heads, seq, head_dim}
    q_heads = Nx.reshape(q_all, {batch_size, seq_len, @num_heads, @head_dim})
    q_heads = Nx.transpose(q_heads, axes: [0, 2, 1, 3]) # {batch, heads, seq, head_dim}

    k_heads = Nx.reshape(k_all, {batch_size, seq_len, @num_heads, @head_dim})
    k_heads = Nx.transpose(k_heads, axes: [0, 2, 1, 3]) # {batch, heads, seq, head_dim}

    v_heads = Nx.reshape(v_all, {batch_size, seq_len, @num_heads, @head_dim})
    v_heads = Nx.transpose(v_heads, axes: [0, 2, 1, 3]) # {batch, heads, seq, head_dim}

    # 3. Matchmaking alignment per head (Q @ K.T)
    # Contracting head_dim axis (3) of both Q and K, batching over batch (0) and head (1) axes → {batch, heads, seq, seq}
    raw_scores = Nx.dot(q_heads, [3], [0, 1], k_heads, [3], [0, 1])

    # 4. Scale
    scale_factor = Nx.sqrt(@head_dim)
    scaled_scores = Nx.divide(raw_scores, scale_factor)

    # 5. Causal masking
    # mask[i, j] = -1.0e9 if j > i, else 0.0
    row_indices = Nx.iota({seq_len, 1})
    col_indices = Nx.iota({1, seq_len})
    mask_condition = Nx.greater(col_indices, row_indices)
    causal_mask = Nx.select(mask_condition, -1.0e9, 0.0) # {seq, seq}
    
    # Broadcast causal mask across batch and head axes
    # causal_mask shape {seq, seq} broadcasts over {batch, heads, seq, seq} via right-aligned broadcasting
    masked_scores = Nx.add(scaled_scores, causal_mask)

    # 6. Softmax
    attention_weights = stable_softmax(masked_scores) # {batch, heads, seq, seq}

    # 7. Value blending per head
    # Contracting seq axis (3) of attention_weights and seq axis (2) of v_heads, batching over batch (0) and head (1) axes → {batch, heads, seq, head_dim}
    attn_out_heads = Nx.dot(attention_weights, [3], [0, 1], v_heads, [2], [0, 1])

    # 8. Recombine heads (Concatenate + Reshape)
    # {batch, heads, seq, head_dim} -> {batch, seq, heads, head_dim} -> {batch, seq, hidden_dim}
    attn_out_transposed = Nx.transpose(attn_out_heads, axes: [0, 2, 1, 3])
    hidden_dim = Nx.axis_size(x, 2)
    attn_out_combined = Nx.reshape(attn_out_transposed, {batch_size, seq_len, hidden_dim})

    # 9. Output Projection
    output = Nx.dot(attn_out_combined, [2], w_o, [0]) # {batch, seq, hidden_dim}
    {output, attention_weights}
  end

  # Gated MLP with SwiGLU activation (Llama style)
  # SwiGLU(x) = (Swish(x W_gate) * x W_up) W_down
  # x: {batch, seq, hidden_dim}
  defn swiglu_mlp(x, w_gate, w_up, w_down) do
    # 1. Gate branch
    gate_branch = Nx.dot(x, [2], w_gate, [0]) # {batch, seq, mlp_dim}
    # Swish = x * sigmoid(x)
    swish_gate = Nx.multiply(gate_branch, Nx.sigmoid(gate_branch))

    # 2. Up projection branch
    up_branch = Nx.dot(x, [2], w_up, [0]) # {batch, seq, mlp_dim}

    # 3. Gated multiplication (elementwise multiply)
    intermediate = Nx.multiply(swish_gate, up_branch) # {batch, seq, mlp_dim}

    # 4. Down projection
    Nx.dot(intermediate, [2], w_down, [0]) # {batch, seq, hidden_dim}
  end

  # Pre-Norm Transformer Block
  defn block_forward(x, gamma1, beta1, w_q, w_k, w_v, w_o, gamma2, beta2, w_gate, w_up, w_down) do
    # --- SUB-LAYER 1: Causal MHA ---
    norm_x1 = Normalization.layer_norm(x, gamma1, beta1)
    {attn_out, weights} = causal_mha(norm_x1, w_q, w_k, w_v, w_o)
    # Additive residual connection: Y = X + Attention(Norm(X))
    y = Nx.add(x, attn_out)

    # --- SUB-LAYER 2: SwiGLU MLP ---
    norm_y1 = Normalization.layer_norm(y, gamma2, beta2)
    mlp_out = swiglu_mlp(norm_y1, w_gate, w_up, w_down)
    # Additive residual connection: Z = Y + MLP(Norm(Y))
    z = Nx.add(y, mlp_out)

    {z, weights, norm_x1, y, norm_y1, mlp_out}
  end
end

# --- RUNNING THE TRANSFORMER BLOCK ---

# Dimensions
_batch = 1
seq = 3
hidden_dim = 8
_num_heads = 2
_head_dim = 4
mlp_dim = 16 # 2x expansion for toy size

# 1. Toy input sequence representing "The cat sat"
# Shape: {batch=1, seq=3, hidden_dim=8}
x = Nx.tensor([[
  [ 1.0,  0.5, -0.2,  0.1,  0.8, -0.4,  0.3,  0.2], # The
  [ 0.1,  2.0,  0.9, -0.3,  0.4,  0.1,  0.5,  0.0], # cat
  [-0.3,  0.2,  1.5,  0.7, -0.2,  0.5,  0.1,  0.8]  # sat
]])

# 2. Normalization parameters (gamma initialized to 1s, beta to 0s)
gamma1 = Nx.broadcast(1.0, {hidden_dim})
beta1 = Nx.broadcast(0.0, {hidden_dim})
gamma2 = Nx.broadcast(1.0, {hidden_dim})
beta2 = Nx.broadcast(0.0, {hidden_dim})

# 3. MHA projection matrices
w_q = Nx.broadcast(0.2, {hidden_dim, hidden_dim})
w_k = Nx.broadcast(0.1, {hidden_dim, hidden_dim})
w_v = Nx.broadcast(0.4, {hidden_dim, hidden_dim})
w_o = Nx.broadcast(0.3, {hidden_dim, hidden_dim})

# 4. SwiGLU MLP projection matrices
w_gate = Nx.broadcast(0.5, {hidden_dim, mlp_dim})
w_up   = Nx.broadcast(0.2, {hidden_dim, mlp_dim})
w_down = Nx.broadcast(0.3, {mlp_dim, hidden_dim})

# Assert that hidden_dim matches num_heads * head_dim
if hidden_dim != 2 * 4, do: raise "hidden_dim #{hidden_dim} must equal num_heads (2) × head_dim (4)"

# Run the compiled pre-norm forward pass
{z, weights, norm_x1, y, norm_y1, mlp_out} =
  TransformerBlock.block_forward(
    x, gamma1, beta1, w_q, w_k, w_v, w_o,
    gamma2, beta2, w_gate, w_up, w_down
  )

# Print shape flows and concrete vector updates
IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 4b: COMPLETE PRE-NORM TRANSFORMER BLOCK WITH CAUSAL MHA & SwiGLU")
IO.puts(String.duplicate("=", 75))
IO.puts("Input Shape X:                  #{inspect(Nx.shape(x))}")
IO.puts("Pre-Attention Norm Shape:       #{inspect(Nx.shape(norm_x1))}")
IO.puts("MHA Attention Weight Matrix:    #{inspect(Nx.shape(weights))}")
IO.puts("First Residual Output Y Shape:  #{inspect(Nx.shape(y))}")
IO.puts("Pre-MLP Norm Shape:             #{inspect(Nx.shape(norm_y1))}")
IO.puts("SwiGLU MLP Output Shape:        #{inspect(Nx.shape(mlp_out))}")
IO.puts("Final Block Output Z Shape:     #{inspect(Nx.shape(z))}\n")

# Demonstrate the stabilizing effect of normalization
IO.puts(String.duplicate("-", 75))
IO.puts("DEMONSTRATING THE STABILIZING EFFECT OF LAYER NORMALIZATION")
IO.puts(String.duplicate("-", 75))

# Compute vector magnitudes (L2 norm) before and after norm
x_flat = Nx.reshape(x, {seq, hidden_dim})
norm_x1_flat = Nx.reshape(norm_x1, {seq, hidden_dim})

# L2 Norm = sqrt(sum(x^2))
x_magnitudes = Nx.sqrt(Nx.sum(Nx.pow(x_flat, 2), axes: [1]))
norm_magnitudes = Nx.sqrt(Nx.sum(Nx.pow(norm_x1_flat, 2), axes: [1]))

IO.puts("Token Vector L2 Magnitudes BEFORE Normalization:")
IO.inspect(x_magnitudes)
IO.puts("Token Vector L2 Magnitudes AFTER Normalization (stabilized to zero-mean/unit-std):")
IO.inspect(norm_magnitudes)
IO.puts("  * Expected post-norm L2 magnitude ≈ √hidden_dim = #{Float.round(:math.sqrt(hidden_dim * 1.0), 3)}")
IO.puts("  * Notice how normalization standardizes the activation variance of every token,")
IO.puts("    preventing activation values from growing or shrinking uncontrollably with depth.\n")

# Show causal attention weights per head
IO.puts(String.duplicate("-", 75))
IO.puts("CAUSAL ATTENTION WEIGHTS (2 HEADS, 3 TOKENS)")
IO.puts(String.duplicate("-", 75))
IO.puts("Attention weights shape: {batch=1, heads=2, seq=3, seq=3}")
IO.inspect(weights)
IO.puts("  * Notice that both Head 0 and Head 1 preserve sequence index boundaries (causality)!")
IO.puts("===========================================================================\n")
