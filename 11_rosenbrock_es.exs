# ===========================================================================
# EXERCISE 1: Evolutionary Optimization on the Rosenbrock curved valley
# ===========================================================================
# In this exercise, we test the capabilities of our Evolution Strategy on a
# highly difficult, non-separable mathematical landscape: the Rosenbrock Function.
# 
# Geometrically, the Rosenbrock function is a steep, curved valley (often called 
# the "banana function"). Finding the valley is easy, but converging to the global
# minimum at [1.0, 1.0] requires the evolutionary population to slowly navigate
# a highly narrow, curved path.

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule RosenbrockObjective do
  import Nx.Defn

  # GPU-Compiled Rosenbrock Function: f(x1, x2) = (1 - x1)^2 + 100 * (x2 - x1^2)^2
  # The global minimum is located at [1.0, 1.0] where f(1.0, 1.0) = 0.0
  defn evaluate(coords) do
    # Extract dimensions using static slices within the compiled graph.
    # `coords[[.., 0]]` reads "all rows, axis-1 index 0" — i.e. column 0 of
    # the {population_size, 2} batch. Verified valid inside `defn` on
    # Nx 0.12.1 (the `..` Access shortcut is supported).
    x1 = coords[[.., 0]]
    x2 = coords[[.., 1]]
    
    # term_1 = (1.0 - x1)^2
    term_1 = Nx.pow(Nx.subtract(1.0, x1), 2)
    # term_2 = 100.0 * (x2 - x1^2)^2
    term_2 = Nx.multiply(100.0, Nx.pow(Nx.subtract(x2, Nx.pow(x1, 2)), 2))
    
    Nx.add(term_1, term_2)
  end
end

defmodule RosenbrockES do
  # Convergence threshold: a best-scout loss below this counts as "we
  # found the [1.0, 1.0] minimum well enough". Used by `Enum.reduce_while`
  # below to halt the loop early instead of always running all 200 gens.
  @convergence_loss 0.01

  def optimize(generations, population_size, mu) do
    # We start our search mean far from the target [1.0, 1.0].
    # Initial coordinate: [-1.0, 2.0]
    mean = Nx.tensor([-1.0, 2.0])
    sigma = 0.5 # Starting search radius

    # Pre-calculate normalized recombination weights
    weights = calculate_weights(mu)

    IO.puts("\n" <> String.duplicate("=", 75))
    IO.puts("EXERCISE 1: NAVIGATING THE CURVED BANANA VALLEY (ROSENBROCK)")
    IO.puts(String.duplicate("=", 75))
    IO.puts("Initial Search Mean Position : #{inspect(Nx.to_flat_list(mean))}")
    IO.puts("Initial Mutation Step (σ)    : #{sigma}")
    IO.puts("Target Global Minimum        : [1.0, 1.0]")
    IO.puts("Convergence threshold        : best_loss < #{@convergence_loss}")
    IO.puts(String.duplicate("=", 75))

    # Use `Enum.reduce_while/3` so we can stop the moment best_loss crosses
    # the convergence threshold. This both prints a clear CONVERGED line
    # AND shaves wall-clock time when the ES happens to find the basin
    # earlier than the worst-case 200 generations.
    final_state =
      Enum.reduce_while(
        1..generations,
        {mean, sigma, 0},
        fn gen, {current_mean, current_sigma, _last_gen} ->
          # Generate unique, isolated random key for population mutations
          mutate_key = Nx.Random.key(gen * 100)

          # Sample Phase: Generate mutated population
          {mutations, _} = Nx.Random.normal(mutate_key, 0.0, 1.0, shape: {population_size, 2})
          population = Nx.add(current_mean, Nx.multiply(current_sigma, mutations))

          # Evaluation Phase on GPU
          losses = RosenbrockObjective.evaluate(population)

          # Selection Phase
          sorted_indices = Nx.argsort(losses) |> Nx.to_flat_list()
          best_indices = Enum.take(sorted_indices, mu)
          best_candidates = Nx.take(population, Nx.tensor(best_indices))

          # Recombination: Weighted average of the top mu candidates
          new_mean = best_candidates
          |> Nx.multiply(weights)
          |> Nx.sum(axes: [0])

          # Step-Size Adaptation: Slow and steady decay to navigate the narrow curve
          new_sigma = if gen > 50, do: current_sigma * 0.98, else: current_sigma

          best_loss = losses[hd(best_indices)] |> Nx.to_number()

          # Log progress
          if rem(gen, 30) == 0 or gen == 1 or gen == generations do
            IO.puts("Gen #{String.pad_leading("#{gen}", 3)} | Best Scout Loss: #{:erlang.float_to_binary(best_loss, [decimals: 5])} | sigma: #{:erlang.float_to_binary(new_sigma, [decimals: 5])} | Current Mean: #{inspect(Nx.to_flat_list(new_mean))}")
          end

          if best_loss < @convergence_loss do
            IO.puts(">> CONVERGED at gen #{gen} with best_loss = #{:erlang.float_to_binary(best_loss, [decimals: 6])} <<")
            {:halt, {new_mean, new_sigma, gen}}
          else
            {:cont, {new_mean, new_sigma, gen}}
          end
        end
      )

    {final_mean, _final_sigma, last_gen} = final_state
    IO.puts("Finished at gen #{last_gen}. Final mean: #{inspect(Nx.to_flat_list(final_mean))}")
    final_state
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

# Run the optimization over 200 generations, population size 40, selecting top 8
# The Rosenbrock valley is curved and requires a larger population and more generations to solve.
RosenbrockES.optimize(200, 40, 8)
