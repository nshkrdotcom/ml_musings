# ===========================================================================
# EXERCISE 1: VISUALIZING THE MOE IMBALANCE PENALTY CURVE
# ===========================================================================
# In this exercise, we simulate various distributions of token loads across
# 4 parallel experts to witness how the Auxiliary Load-Balancing Loss scales.
#
# We evaluate distributions from completely collapsed (all tokens to 1 expert)
# to perfectly balanced, plotting a text-based ASCII scale of the penalty.

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule GatingLossSimulator do
  import Nx.Defn

  # GPU-Compiled Auxiliary Loss calculation
  # L_aux = E * sum(f_i * P_i)
  defn calculate_aux_loss(f_i, p_i, num_experts) do
    dot_product = Nx.sum(Nx.multiply(f_i, p_i))
    Nx.multiply(num_experts, dot_product)
  end

  def run_simulations() do
    # Use a float here because calculate_aux_loss/3 multiplies by it inside a defn,
    # and Nx.multiply works most naturally with matching float tensor types.
    num_experts_f = 4.0

    # Define various load distributions to simulate
    # Format: {label, f_i, p_i}
    scenarios = [
      {"PERFECTLY BALANCED (Ideal Parallel Load)", 
       [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]},
       
      {"SLIGHTLY SKEWED (Mild imbalance)", 
       [0.40, 0.20, 0.20, 0.20], [0.40, 0.20, 0.20, 0.20]},
       
      {"MODERATELY SKEWED (Two idle experts)", 
       [0.50, 0.50, 0.00, 0.00], [0.50, 0.50, 0.00, 0.00]},
       
      {"HEAVILY SKEWED (Severe imbalance)", 
       [0.70, 0.10, 0.10, 0.10], [0.70, 0.10, 0.10, 0.10]},
       
      {"TOTAL COLLAPSE (Single overloaded expert)", 
       [1.00, 0.00, 0.00, 0.00], [1.00, 0.00, 0.00, 0.00]}
    ]

    IO.puts("\n" <> String.duplicate("=", 75))
    IO.puts("EXERCISE 1: VISUALIZING THE GATING IMBALANCE PENALTY CURVE (4 EXPERTS)")
    IO.puts(String.duplicate("=", 75))
    IO.puts("Auxiliary Loss Bounds:")
    IO.puts("  - Absolute Minimum (Perfect Balance) : 1.0")
    IO.puts("  - Absolute Maximum (Total Collapse)  : 4.0")
    IO.puts(String.duplicate("=", 75))

    Enum.each(scenarios, fn {label, f_list, p_list} ->
      f_tensor = Nx.tensor(f_list)
      p_tensor = Nx.tensor(p_list)
      
      # Calculate loss on GPU
      loss_tensor = calculate_aux_loss(f_tensor, p_tensor, num_experts_f)
      loss_val = Nx.to_number(loss_tensor)

      # Generate ASCII bar plot to visualize the penalty curve
      # Bar spans from 1.0 (min) to 4.0 (max)
      bar_width = 30
      normalized_val = (loss_val - 1.0) / 3.0 # scale between 0.0 and 1.0
      filled_chars = round(normalized_val * bar_width)
      empty_chars = bar_width - filled_chars
      
      bar = String.duplicate("█", filled_chars) <> String.duplicate("░", empty_chars)

      IO.puts("\nScenario: #{label}")
      IO.puts("  f_i (Load Fraction)    : #{inspect(f_list)}")
      IO.puts("  P_i (Avg Probability)  : #{inspect(p_list)}")
      IO.puts("  Auxiliary Loss Value   : #{Float.round(loss_val, 4)}")
      IO.puts("  Penalty Level          : [#{bar}]")
    end)
    IO.puts(String.duplicate("=", 75))
  end
end

GatingLossSimulator.run_simulations()
