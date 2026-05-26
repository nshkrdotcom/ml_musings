# ===========================================================================
# LESSON 6: Mixture of Experts (MoE) & Gating Load Balancing
# ===========================================================================
# In this lesson, we explore the Mixture of Experts (MoE) architecture, 
# focusing on sparse routing and the critical threat of "Expert Collapse".
#
# To prevent the gating router from overloading a single expert while others 
# sit completely idle, we implement an Auxiliary Load-Balancing Loss from scratch 
# on the GPU using Numerical Elixir (Nx).

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule MoERouter do
  import Nx.Defn

  defn stable_softmax(t) do
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    t_shifted = Nx.subtract(t, max_vals)
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    Nx.divide(exps, sum_exps)
  end

  # Use defn to compile our routing logic and loss calculation to GPU kernels
  defn route_and_calculate_loss(x, w_g) do
    num_tokens = Nx.axis_size(x, 0)
    num_experts = Nx.axis_size(w_g, 0)

    # Step 1: Calculate raw routing logits: S = X . W_g^T
    # x is shape {N, D}, w_g is shape {E, D}. logits becomes {N, E}.
    logits = Nx.dot(x, [1], w_g, [1])

    # Step 2: Calculate soft gating probabilities (P_i)
    # Applying softmax across experts axis (axis 1) for each token
    probs = stable_softmax(logits)
    
    # P_i: Average probability assigned to expert i across the entire batch
    p_i = Nx.mean(probs, axes: [0])

    # Step 3: Top-1 hard routing decision
    # Find the index of the highest scoring expert for each token
    selected_expert_indices = Nx.argmax(logits, axis: 1)

    # Step 4: Calculate the fraction of tokens routed to each expert (f_i)
    # Create a one-hot encoding of the selected expert indices by broadcasting
    one_hot_selections = Nx.equal(
      Nx.reshape(selected_expert_indices, {num_tokens, 1}),
      Nx.iota({1, num_experts})
    )
    
    # f_i: Fraction of tokens routed to expert i (sum one-hot along tokens axis)
    f_i = Nx.divide(Nx.sum(one_hot_selections, axes: [0]), num_tokens)

    # Step 5: Compute the Auxiliary Load Balancing Loss: L_aux = E * sum(f_i * P_i)
    dot_product = Nx.sum(Nx.multiply(f_i, p_i))
    aux_loss = Nx.multiply(num_experts, dot_product)

    {selected_expert_indices, f_i, p_i, aux_loss}
  end
end

# --- RUNNING THE SYSTEMS COMPARISON ---

# Input Batch: 6 tokens, each 4-dimensional (representing token embeddings)
x = Nx.tensor([
  [1.0, 0.1, 0.2, 0.1],  # Token 1
  [0.1, 1.2, 0.1, 0.2],  # Token 2
  [0.2, 0.1, 1.5, 0.1],  # Token 3
  [1.1, 0.2, 0.1, 0.3],  # Token 4
  [0.1, 1.0, 0.3, 0.1],  # Token 5
  [0.3, 0.2, 1.4, 0.2]   # Token 6
])

# Scenario A: COLLAPSED ROUTER WEIGHTS
# Gating weights are heavily biased towards Expert 0 (Expert Collapse simulation)
w_g_collapsed = Nx.tensor([
  [ 2.0,  2.0,  2.0,  2.0],  # Expert 0 (Strong starting weights)
  [-1.0, -1.0, -1.0, -1.0],  # Expert 1
  [-2.0, -2.0, -2.0, -2.0]   # Expert 2
])

# Scenario B: BALANCED ROUTER WEIGHTS
# Router weights are specialized to route different coordinate dimensions to different experts
w_g_balanced = Nx.tensor([
  [ 2.0, -1.0, -1.0,  0.0],  # Expert 0 (Specialized for Dimension 0 / Tokens 1 & 4)
  [-1.0,  2.0, -1.0,  0.0],  # Expert 1 (Specialized for Dimension 1 / Tokens 2 & 5)
  [-1.0, -1.0,  2.0,  0.0]   # Expert 2 (Specialized for Dimension 2 / Tokens 3 & 6)
])

IO.puts(String.duplicate("=", 75))
IO.puts("LESSON 6: mixture of experts (moe) sparse routing & load balancing")
IO.puts(String.duplicate("=", 75))

# Scenario A: Collapsed Router
IO.puts("EVALUATING SCENARIO A: COLLAPSED ROUTER (ALL TOKENS OVERLOAD EXP 0)")
{selections_a, f_a, p_a, loss_a} = MoERouter.route_and_calculate_loss(x, w_g_collapsed)

IO.puts("Token Expert Selections  : #{inspect(Nx.to_flat_list(selections_a))}")
IO.puts("Fraction of load (f_i)    : #{inspect(Nx.to_flat_list(f_a))}")
IO.puts("Avg probabilities (P_i)  : #{inspect(Nx.to_flat_list(p_a))}")
IO.puts("Auxiliary Loss (theoretical max for E=3 is 3.0) : #{Float.round(Nx.to_number(loss_a), 5)}")

IO.puts(String.duplicate("-", 75))

# Scenario B: Balanced Router
IO.puts("EVALUATING SCENARIO B: BALANCED ROUTER (PERFECTLY DISTRIBUTED WORK)")
{selections_b, f_b, p_b, loss_b} = MoERouter.route_and_calculate_loss(x, w_g_balanced)

IO.puts("Token Expert Selections  : #{inspect(Nx.to_flat_list(selections_b))}")
IO.puts("Fraction of load (f_i)    : #{inspect(Nx.to_flat_list(f_b))}")
IO.puts("Avg probabilities (P_i)  : #{inspect(Nx.to_flat_list(p_b))}")
IO.puts("Auxiliary Loss (theoretical min for E=3 is 1.0) : #{Float.round(Nx.to_number(loss_b), 5)}")
IO.puts(String.duplicate("=", 75))
