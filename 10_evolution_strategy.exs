# ===========================================================================
# LESSON 5: Black-Box Optimization vs. White-Box Gradients
# ===========================================================================
# In this lesson, we explore black-box, derivative-free optimization (Evolution
# Strategy) and compare it side-by-side with gradient-based optimization
# (Gradient Descent) on a noisy landscape.
# 
# While neural network weights are updated using backpropagation (Gradient Descent),
# coordinating networks must sometimes route work to external black-box systems
# (e.g., calling discrete LLM APIs) where gradients cannot flow.
#
# We optimize a noisy Sphere landscape: f(x1, x2) = x1^2 + x2^2 + noise,
# starting from [5.0, -5.0], comparing ES and GD side-by-side under identical
# noise conditions.

# Dynamically pull in Hex dependencies
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule NoisyObjective do
  import Nx.Defn

  # GPU-Compiled Noisy Sphere Landscape: f(x1, x2) = x1^2 + x2^2 + noise
  defn evaluate(coords, noise_key) do
    {noise, _} = Nx.Random.normal(noise_key, 0.0, 0.1, shape: Nx.shape(coords))
    
    distance = coords
    |> Nx.pow(2)
    |> Nx.sum(axes: [1])
    
    Nx.add(distance, Nx.sum(noise, axes: [1]))
  end
end

defmodule SphereOptimizer do
  import Nx.Defn

  # GPU-Compiled Gradient Descent step using value_and_grad
  defn gd_step(coords, noise_key, lr) do
    {loss_val, {grad, _grad_key}} = value_and_grad({coords, noise_key}, fn {c, k} ->
      c_reshaped = Nx.reshape(c, {1, 2})
      loss_arr = NoisyObjective.evaluate(c_reshaped, k)
      Nx.reshape(loss_arr, {})
    end)

    new_coords = Nx.subtract(coords, Nx.multiply(lr, grad))
    {new_coords, loss_val}
  end

  # Orchestrating loop running both ES and GD side-by-side
  def optimize(generations, population_size, mu) do
    # Initialize search starting position far away: [5.0, -5.0]
    es_mean = Nx.tensor([5.0, -5.0])
    gd_coords = Nx.tensor([5.0, -5.0])
    sigma = 2.0
    lr = Nx.tensor(0.1)

    weights = calculate_weights(mu)

    IO.puts("\n" <> String.duplicate("=", 85))
    IO.puts("LESSON 5: EVOLUTION STRATEGY (BLACK-BOX) VS. GRADIENT DESCENT (WHITE-BOX) ON GPU")
    IO.puts(String.duplicate("=", 85))
    IO.puts("Initial Start Position       : [5.0, -5.0]")
    IO.puts("ES Mutation Step (σ)         : #{sigma}")
    IO.puts("GD Learning Rate (lr)        : 0.1")
    IO.puts("Noise Standard Deviation     : 0.1 (effective noise per scout ~= 0.14)")
    IO.puts(String.duplicate("=", 85))

    Enum.reduce(1..generations, {es_mean, sigma, gd_coords}, fn gen, {current_mean, current_sigma, current_gd} ->
      noise_key = Nx.Random.key(gen * 100)
      mutate_key = Nx.Random.key(gen * 100 + 1)

      # --- 1. EVOLUTION STRATEGY STEP ---
      {mutations, _} = Nx.Random.normal(mutate_key, 0.0, 1.0, shape: {population_size, 2})
      population = Nx.add(current_mean, Nx.multiply(current_sigma, mutations))
      losses = NoisyObjective.evaluate(population, noise_key)
      sorted_indices = Nx.argsort(losses) |> Nx.to_flat_list()
      best_indices = Enum.take(sorted_indices, mu)
      best_candidates = Nx.take(population, Nx.tensor(best_indices))
      new_mean = best_candidates |> Nx.multiply(weights) |> Nx.sum(axes: [0])
      new_sigma = if gen > 30, do: current_sigma * 0.95, else: current_sigma
      best_es_loss = losses[hd(best_indices)] |> Nx.to_number()

      # --- 2. GRADIENT DESCENT STEP ---
      # GD uses the exact same noise key to ensure perfectly fair comparisons
      {new_gd, gd_loss_val} = gd_step(current_gd, noise_key, lr)
      gd_loss = Nx.to_number(gd_loss_val)

      # Log progress side-by-side
      if rem(gen, 10) == 0 or gen == 1 do
        es_pos_str = Enum.map(Nx.to_flat_list(new_mean), &Float.round(&1, 3)) |> inspect()
        gd_pos_str = Enum.map(Nx.to_flat_list(new_gd), &Float.round(&1, 3)) |> inspect()

        IO.puts("Gen #{String.pad_leading("#{gen}", 2)} | " <>
                "ES Loss: #{:erlang.float_to_binary(best_es_loss, [decimals: 5])} (Mean: #{es_pos_str}) | " <>
                "GD Loss: #{:erlang.float_to_binary(gd_loss, [decimals: 5])} (Pos: #{gd_pos_str})")
      end

      {new_mean, new_sigma, new_gd}
    end)

    IO.puts(String.duplicate("=", 85))
    IO.puts("OPTIMIZATION COMPARISON COMPLETE!")
    IO.puts("  * Observe how Gradient Descent converges extremely fast because it has direct access")
    IO.puts("    to slope/direction information (analytical gradients).")
    IO.puts("  * Observe how the Evolution Strategy successfully converges to the optimum despite")
    IO.puts("    having ZERO access to derivatives, purely by sampling coordinate deviations!")
    IO.puts(String.duplicate("=", 85) <> "\n")
  end

  defp calculate_weights(mu) do
    raw_weights = Enum.map(1..mu, fn i -> :math.log(mu + 0.5) - :math.log(i) end)
    sum = Enum.sum(raw_weights)
    
    raw_weights
    |> Enum.map(fn w -> w / sum end)
    |> Nx.tensor()
    |> Nx.reshape({mu, 1})
  end
end

SphereOptimizer.optimize(80, 20, 5)
