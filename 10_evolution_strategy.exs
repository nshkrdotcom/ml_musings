# ===========================================================================
# LESSON 5: Black-Box Optimization (The Evolution Strategy)
# ===========================================================================
# In this lesson, we explore black-box, derivative-free optimization.
# 
# While neural network weights are updated using backpropagation, coordinating
# networks (like TRINITY) must route work to external APIs (Claude, GPT, Gemini).
# These APIs are non-differentiable black boxes. We cannot compute a gradient.
#
# To solve this, we implement a complete Evolution Strategy (ES) from scratch.
# We optimize a noisy, non-differentiable Sphere landscape, finding the origin
# [0.0, 0.0] without ever calculating a mathematical slope (gradient).

# Dynamically pull in the latest stable packages from Hex
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA backend to compile all computations directly to CUDA GPU kernels
Nx.global_default_backend(EXLA.Backend)

defmodule NoisyObjective do
  import Nx.Defn

  # GPU-Compiled Noisy Sphere Landscape: f(x1, x2) = x1^2 + x2^2 + noise
  # We add random Gaussian noise to simulate the noisy, high-variance evaluations
  # that occur when calling discrete LLM text APIs.
  defn evaluate(coords, noise_key) do
    {noise, _} = Nx.Random.normal(noise_key, 0.0, 0.1, shape: Nx.shape(coords))
    
    # NOTE: we generate one noise sample PER coordinate axis (shape == coords.shape),
    # then sum the noise along the feature axis below. Because both feature dimensions
    # contribute independent Normal(0, 0.1) noise, the effective per-scout evaluation
    # noise has standard deviation 0.1 * sqrt(num_features) = 0.1 * sqrt(2) ~= 0.14.
    # Sphere function: sum along features of squared coordinates
    distance = coords
    |> Nx.pow(2)
    |> Nx.sum(axes: [1])
    
    # Add noise to simulate real-world evaluation variance
    Nx.add(distance, Nx.sum(noise, axes: [1]))
  end
end

defmodule SphereES do
  # We implement the orchestrating loop in standard Elixir. 
  # This is a realistic design: evaluating the scouts (e.g. hitting APIs or 
  # running tasks) is an orchestrator-level execution.
  def optimize(generations, population_size, mu) do
    # 1. Initialize our search distribution Mean (m) and Step Size (sigma).
    # We start our mean far away from the origin: [5.0, -5.0].
    mean = Nx.tensor([5.0, -5.0])

    # SIGMA TUNING RULE OF THUMB: pick sigma so that
    #   distance_from_optimum / sigma  ≈  3-5 standard deviations.
    # Here the L2 distance from start [5.0, -5.0] to the origin is
    # sqrt(50) ≈ 7.07. With sigma = 2.0 that puts the optimum
    # ~3.5σ away, which is comfortably reachable by Gaussian mutations
    # within the first few generations. (The old sigma = 1.0 put the
    # optimum ~7σ away and required many generations just to graze
    # the basin.)
    sigma = 2.0

    # Pre-calculate normalized recombination weights favoring top-performing scouts
    weights = calculate_weights(mu)

    IO.puts("\n" <> String.duplicate("=", 75))
    IO.puts("LESSON 5: DERIVATIVE-FREE BLACK-BOX EVOLUTION STRATEGY ON GPU")
    IO.puts(String.duplicate("=", 75))
    IO.puts("Initial Search Mean Position : #{inspect(Nx.to_flat_list(mean))}")
    IO.puts("Initial Mutation Step (σ)    : #{sigma}")
    IO.puts(String.duplicate("=", 75))

    Enum.reduce(1..generations, {mean, sigma}, fn gen, {current_mean, current_sigma} ->
      # Generate unique, isolated random keys for evaluation noise and population mutations
      noise_key = Nx.Random.key(gen * 100)
      mutate_key = Nx.Random.key(gen * 100 + 1)

      # 2. SAMPLE PHASE: Mutate the current mean using random Gaussian noise (Scouts)
      {mutations, _} = Nx.Random.normal(mutate_key, 0.0, 1.0, shape: {population_size, 2})
      population = Nx.add(current_mean, Nx.multiply(current_sigma, mutations))

      # 3. EVALUATION PHASE: Measure the fitness (loss) of each candidate in the black box
      losses = NoisyObjective.evaluate(population, noise_key)

      # 4. SELECTION PHASE: Sort the candidates by performance (lowest loss is best).
      # NOTE: `best_indices` is non-empty as long as `mu >= 1` and `population_size >= mu`,
      # which the call site below guarantees. If you ever set `mu = 0`, the
      # `hd(best_indices)` lookup further down will crash with a confusing
      # `Enum.EmptyError`-style message — keep `mu >= 1`.
      sorted_indices = Nx.argsort(losses) |> Nx.to_flat_list()
      best_indices = Enum.take(sorted_indices, mu)

      # Extract the top performing scouts
      best_candidates = Nx.take(population, Nx.tensor(best_indices))

      # 5. RECOMBINATION PHASE: Update the mean as a weighted average of the top scouts
      new_mean = best_candidates
      |> Nx.multiply(weights)
      |> Nx.sum(axes: [0])

      # Simple Step-Size Adaptation: Shrink search radius as we get closer to focus the search
      new_sigma = if gen > 30, do: current_sigma * 0.95, else: current_sigma

      # Log progress at intervals
      if rem(gen, 10) == 0 or gen == 1 do
        # Accessing losses using scalar indexing [] returns a 0D scalar tensor, compatible with Nx.to_number/1
        best_loss = losses[hd(best_indices)] |> Nx.to_number()
        IO.puts("Gen #{String.pad_leading("#{gen}", 2)} | Best Scout Loss: #{:erlang.float_to_binary(best_loss, [decimals: 5])} | sigma: #{:erlang.float_to_binary(new_sigma, [decimals: 5])} | Current Mean: #{inspect(Nx.to_flat_list(new_mean))}")
      end

      {new_mean, new_sigma}
    end)
  end

  defp calculate_weights(mu) do
    # Create normalized weights that sum to 1.0, favoring top-performing scouts
    # Formula: w_i = log(mu + 0.5) - log(i)
    raw_weights = Enum.map(1..mu, fn i -> :math.log(mu + 0.5) - :math.log(i) end)
    sum = Enum.sum(raw_weights)
    
    raw_weights
    |> Enum.map(fn w -> w / sum end)
    |> Nx.tensor()
    |> Nx.reshape({mu, 1})
  end
end

# Run the optimization over 80 generations, population size 20, selecting top 5
SphereES.optimize(80, 20, 5)
