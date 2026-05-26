# ===========================================================================
# LESSON 2: Linear Probing and Representational Geometry
# ===========================================================================
# In this lesson, we build a Linear Probe from scratch. Geometrically, a probe
# is a separating hyperplane that cuts through representation space to divide
# different categories of semantic vectors (e.g. "Math" vs. "Writing").
#
# We generate synthetic 2D vector clusters, compile an optimization loop using
# Numerical Elixir (Nx) automatic differentiation (grad), train the probe,
# and evaluate its accuracy on unseen, out-of-sample data.

# Dynamically pull in the latest stable dependencies from Hex
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA backend to compile all computations directly to CUDA GPU kernels
Nx.global_default_backend(EXLA.Backend)

defmodule SyntheticData do
  @doc """
  Generates random clusters of coordinates in a 2D space.
  - Class 0 (Math tasks) center around [2.0, 2.0]
  - Class 1 (Writing tasks) center around [-2.0, -2.0]
  We accept a seed parameter so we can generate distinct training and validation sets.
  """
  def generate(num_samples, seed) do
    key = Nx.Random.key(seed)
    half = div(num_samples, 2)

    # Class 0: Math tasks
    {noise_0, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {half, 2})
    class_0 = Nx.add(noise_0, Nx.tensor([2.0, 2.0]))
    labels_0 = Nx.broadcast(0.0, {half, 1})

    # Class 1: Writing tasks
    {noise_1, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {half, 2})
    class_1 = Nx.add(noise_1, Nx.tensor([-2.0, -2.0]))
    labels_1 = Nx.broadcast(1.0, {half, 1})

    # Concatenate clusters into a single dataset
    x = Nx.concatenate([class_0, class_1], axis: 0)
    y = Nx.concatenate([labels_0, labels_1], axis: 0)

    {x, y}
  end
end

defmodule LinearProbe do
  import Nx.Defn

  # L2 weight-decay coefficient. Small value: enough to bias the probe toward
  # simpler separating hyperplanes without dominating the gradient signal.
  #
  # NOTE: this module attribute is interpolated to the float literal 0.001
  # at the Elixir compile site (NOT inside defn), and then passed to
  # `update/6` and `loss/5` as an explicit defn parameter (see the call
  # site below). That means lambda flows through defn as a scalar tensor
  # argument with a stable type/shape, and there is NO per-call XLA
  # recompilation. (If you ever switch lambda to a value that changes
  # between calls, make sure it still arrives as the same scalar dtype to
  # keep the compiled cache warm.)
  @lambda 0.001

  # 1. MODEL PREDICTION FORMULA (z = w · x + b)
  # Computes dot product of inputs x and weights w, adds the bias scalar b,
  # and applies the Sigmoid function to squash scores into [0.0, 1.0] probabilities.
  defn predict(w, b, x) do
    # Dot product along the feature axes: [1] in x corresponds to [0] in w
    z = Nx.add(Nx.dot(x, [1], w, [0]), b)
    Nx.sigmoid(z)
  end

  # 2. BINARY CROSS ENTROPY LOSS FUNCTION
  # Measures the error (entropy) between prediction probabilities and true labels.
  # We add a tiny epsilon (1.0e-7) directly inside the log function to prevent
  # log(0) NaN errors without breaking the XLA chain rule autodiff compiler.
  defn loss(w, b, x, y, lambda) do
    preds = predict(w, b, x)

    term_1 = Nx.multiply(y, Nx.log(Nx.add(preds, 1.0e-7)))
    term_2 = Nx.multiply(Nx.subtract(1.0, y), Nx.log(Nx.add(Nx.subtract(1.0, preds), 1.0e-7)))

    bce = Nx.mean(Nx.negate(Nx.add(term_1, term_2)))

    # L2 weight-decay penalty (sum of squared weights, scaled by lambda).
    # Lesson 1 notes warn that high-dimensional separability is "free", so a
    # linear probe is at risk of fitting noise. L2 regularization shrinks the
    # separating hyperplane normal toward the origin, biasing the probe toward
    # the simpler hypothesis when several decision boundaries fit the data.
    l2_penalty = Nx.multiply(lambda, Nx.sum(Nx.pow(w, 2)))

    Nx.add(bce, l2_penalty)
  end

  # 3. GRADIENT DESCENT UPDATE STEP (Nx Automatic Differentiation)
  # Uses value_and_grad/2 to compute both the current loss value and the
  # exact partial derivatives (gradients) of the loss with respect to w and b.
  defn update(w, b, x, y, learning_rate, lambda) do
    {loss_val, {grad_w, grad_b}} = value_and_grad({w, b}, fn {w_arg, b_arg} ->
      loss(w_arg, b_arg, x, y, lambda)
    end)

    # Adjust parameters slightly in the OPPOSITE direction of the gradient
    new_w = Nx.subtract(w, Nx.multiply(learning_rate, grad_w))
    new_b = Nx.subtract(b, Nx.multiply(learning_rate, grad_b))

    {new_w, new_b, loss_val}
  end

  # 4. OUT-OF-SAMPLE ACCURACY EVALUATOR (Exercise 1)
  # Runs inference on unseen validation datasets and computes classification accuracy.
  # CONTRACT: returns accuracy AS A PERCENTAGE in [0.0, 100.0] (the final
  # `Nx.multiply(100.0)` is the unit-conversion step). The Wilson-CI code
  # below depends on this percentage convention.
  defn evaluate(w, b, x_val, y_val) do
    preds = predict(w, b, x_val)
    # Threshold predictions at 0.5: >= 0.5 is Class 1 (Writing), < 0.5 is Class 0 (Math)
    binary_preds = Nx.greater_equal(preds, 0.5)
    matches = Nx.equal(binary_preds, y_val)
    
    # Compute accuracy percentage as the mean of matching boolean values
    Nx.mean(Nx.as_type(matches, {:f, 32})) |> Nx.multiply(100.0)
  end

  # Helper to print a beautiful 2D ASCII scatter plot of the descending training loss curve
  def print_ascii_plot(history) do
    max_loss = Enum.max(history)
    min_loss = Enum.min(history)
    loss_range = max_loss - min_loss
    safe_range = max(loss_range, 1.0e-8)

    # Height of plot in lines, width in characters
    height = 12
    width = 50

    # Resample history to exactly `width` points for clean plotting
    chunk_size = max(1, div(length(history), width))
    sampled = 
      history
      |> Enum.chunk_every(chunk_size)
      |> Enum.map(& (Enum.sum(&1) / length(&1)))
      |> Enum.take(width)

    IO.puts("\nTRAINING LOSS CURVE (ASCII VISUALIZATION):")
    IO.puts(String.duplicate("-", 75))
    
    # Loop from top row down to 0
    Enum.each((height - 1)..0//-1, fn row ->
      label = 
        cond do
          row == height - 1 ->
            :io_lib.format("~.4f | ", [max_loss]) |> List.to_string()
          row == div(height, 2) ->
            mid_val = min_loss + 0.5 * loss_range
            :io_lib.format("~.4f | ", [mid_val]) |> List.to_string()
          row == 0 ->
            :io_lib.format("~.4f | ", [min_loss]) |> List.to_string()
          true ->
            "       | "
        end

      row_str =
        sampled
        |> Enum.map(fn val ->
          row_bucket = round((val - min_loss) / safe_range * (height - 1))
          if row_bucket == row, do: "●", else: " "
        end)
        |> Enum.join("")

      IO.puts(label <> row_str)
    end)
    
    # Axis bottom
    IO.puts("       +" <> String.duplicate("-", width))
    
    # Bottom labels for epochs
    total_epochs = length(history)
    label_step = div(width, 4)
    epoch_step = div(total_epochs, 4)
    
    # label alignment assumes epoch numbers ≤ 3 digits; adjust label_step arithmetic for longer runs
    labels = 
      for i <- 0..4 do
        epoch_num = i * epoch_step
        {inspect(epoch_num), i * label_step}
      end

    bottom_line =
      Enum.reduce(labels, String.duplicate(" ", width + 10), fn {str, pos}, acc ->
        start_idx = pos + 8
        len = String.length(str)
        String.slice(acc, 0, start_idx) <> str <> String.slice(acc, start_idx + len, String.length(acc))
      end)
      
    IO.puts(bottom_line)
    IO.puts(String.duplicate("-", 75))
  end

  # Orchestrator Training & Validation Loop
  def run_curriculum(epochs, lr) do
    # Generate Training dataset (Seed 42)
    {x_train, y_train} = SyntheticData.generate(1000, 42)

    # If SyntheticData.generate produces BinaryBackend tensors, update/6 will silently copy to device each call.

    # Generate Unseen Out-of-Sample Validation dataset (Seed 999 for Exercise 1)
    {x_val, y_val} = SyntheticData.generate(500, 999)

    # Initialize random weights (2 inputs -> 1 output) and zero bias
    w_key = Nx.Random.key(100)
    {w_init, _} = Nx.Random.normal(w_key, 0.0, 1.0, shape: {2, 1})
    b_init = Nx.tensor([[0.0]])

    IO.puts("\n" <> String.duplicate("=", 75))
    IO.puts("LESSON 2: TRAINING A GEOMETRIC LINEAR PROBE FROM SCRATCH")
    IO.puts(String.duplicate("=", 75))
    IO.puts("L2 weight-decay coefficient (lambda): #{@lambda}")
    IO.puts("Initial Weights (Separating Line Normal):\n#{inspect(w_init)}")
    IO.puts("Initial Bias (Separating Line Offset):\n#{inspect(b_init)}")
    IO.puts(String.duplicate("=", 75))

    # Perform gradient descent training loop, collecting losses
    {final_w, final_b, loss_history_rev} = 
      Enum.reduce(1..epochs, {w_init, b_init, []}, fn epoch, {current_w, current_b, history} ->
        {next_w, next_b, loss_val} = update(current_w, current_b, x_train, y_train, lr, @lambda)
        loss_num = Nx.to_number(loss_val)

        if rem(epoch, 20) == 0 do
          # Evaluate validation accuracy at intermediate steps on device
          val_acc = evaluate(next_w, next_b, x_val, y_val) |> Nx.to_number()
          IO.puts("Epoch #{String.pad_leading("#{epoch}", 3)} | Training Loss: #{:erlang.float_to_binary(loss_num, [decimals: 5])} | Val Accuracy: #{:erlang.float_to_binary(val_acc, [decimals: 2])}%")
        end

        {next_w, next_b, [loss_num | history]}
      end)

    loss_history = Enum.reverse(loss_history_rev)
    print_ascii_plot(loss_history)

    IO.puts(String.duplicate("=", 75))
    IO.puts("TRAINING COMPLETE!")
    IO.puts(String.duplicate("=", 75))
    IO.puts("Final Weights (Separating Line Normal):\n#{inspect(final_w)}")
    IO.puts("Final Bias (Separating Line Offset):\n#{inspect(final_b)}")

    # ---------------------------------------------------------------------------
    # EXERCISE 1 OUT-OF-SAMPLE VERIFICATION
    # ---------------------------------------------------------------------------
    final_val_acc = evaluate(final_w, final_b, x_val, y_val) |> Nx.to_number()
    IO.puts("\n" <> String.duplicate("-", 75))
    IO.puts("EXERCISE 1 VERIFICATION (Validation on Unseen Seed 999):")
    IO.puts("Validation Accuracy on Unseen Dataset: #{:erlang.float_to_binary(final_val_acc, [decimals: 2])}%")
    
    # ---------------------------------------------------------------------------
    # WILSON 95% CONFIDENCE INTERVAL ON THE VALIDATION ACCURACY
    # ---------------------------------------------------------------------------
    # Instead of a hard-coded "95% threshold" (which is statistically arbitrary),
    # we report the Wilson score interval for a binomial proportion. The Wilson
    # interval gives an asymmetric range [lower, upper] that, under the binomial
    # model, contains the true success probability with ~95% confidence.
    n_val = 500
    # `evaluate/4` returns accuracy as a percentage in [0.0, 100.0] (see its
    # CONTRACT note), so we divide by 100.0 to convert into a probability
    # `p_hat` in [0.0, 1.0]. The Wilson formula expects probability units.
    p_hat = final_val_acc / 100.0
    z = 1.96
    denom = 1.0 + z * z / n_val
    center = (p_hat + z * z / (2.0 * n_val)) / denom
    margin = (z * :math.sqrt(p_hat * (1.0 - p_hat) / n_val + z * z / (4.0 * n_val * n_val))) / denom
    # Convert back to percentage units for display; clamp to [0%, 100%]
    # since the Wilson bounds can technically exceed [0, 1] before clamping.
    lower = max(0.0, (center - margin) * 100.0)
    upper = min(100.0, (center + margin) * 100.0)
    IO.puts("Wilson 95% CI for validation accuracy: " <>
      "[#{:erlang.float_to_binary(lower, [decimals: 2])}%, " <>
      "#{:erlang.float_to_binary(upper, [decimals: 2])}%] " <>
      "(N = #{n_val})")
    IO.puts(String.duplicate("-", 75))
  end
end

# Train the probe for 100 epochs with a learning rate of 0.1
LinearProbe.run_curriculum(100, 0.1)
