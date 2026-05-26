# ===========================================================================
# CAPSTONE LESSON: Build a Mini-TRINITY Governed Execution Substrate
# ===========================================================================
# Congratulations on reaching the Capstone! You have navigated the mathematical,
# geometric, and computational building blocks undergirding modern AI systems:
#
#   - LESSON 1: High-dimensional geometry, vector spaces, and quasi-orthogonality.
#   - LESSON 2: Differentiable representation spaces and classifying hyperplanes.
#   - LESSON 3: Self-Attention routing matrices (Queries, Keys, and Values).
#   - LESSON 4: SVD surgery and Low-Rank Adaptation (LoRA) parameter tuning.
#   - LESSON 5: Derivative-free black-box optimization (Evolution Strategies).
#   - LESSON 6: Sparse Mixture of Experts (MoE) routing and balancing.
#
# Today, we compile these components into a single, cohesive, working framework:
# The Mini-TRINITY Substrate. This implements a stateful, governed, closed-loop 
# write-path that routes, executes, verifies, and dynamically escalates/rewarps 
# execution context based on real-time feedback signals.
#
# ===========================================================================
# HOW THE LESSONS INTEGRATE INTO THIS SUBSTRATE:
#
# 1. THE REPRESENTATION MANIFOLD (Lessons 1 & 2):
#    User intents are ingested as 2-dimensional coordinate vectors. Their geometric
#    position determines their semantic alignment (e.g., Math quadrant, Creative quadrant).
#
# 2. THE ROUTER HEAD (Lesson 6):
#    A compiled single-layer projection matrix maps our task vectors directly to routing
#    probabilities across 3 Experts. This is compiled natively on the GPU using `defn`.
#
# 3. CLOSED-LOOP FEEDBACK & COORDINATE WARPING (Lessons 3 & 5):
#    If the verifier rejects an expert's output, we cannot backpropagate through the
#    non-differentiable expert boundary (e.g., third-party APIs). Instead, we apply
#    a control-theory coordinate warp: we shift the context vector *away* from the
#    failed expert's region of the manifold. This forces the router to escalate to a
#    stronger expert on the next turn.
# ===========================================================================

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Configure EXLA for compiled math execution on the GPU or CPU fallback
Nx.global_default_backend(EXLA.Backend)

defmodule MiniTrinity.Router do
  @moduledoc """
  Compiled routing projector representing our Gating Network (Lesson 6).
  It evaluates task representations and projects them into expert routing scores.
  """
  import Nx.Defn

  @doc """
  Projects a 2D task coordinate vector into a probability distribution over 3 Experts.
  Runs as a compiled computation graph directly on the hardware device.
  """
  defn select_expert(task_vector, routing_weights) do
    # Dot product: Logits = Task . Weights^T (Lesson 2 classification hyperplane).
    #
    # SHAPES: task_vector has shape {D=2}, routing_weights has shape {E=3, D=2}.
    # We contract axis 0 of `task_vector` (the only axis it has — the feature
    # axis) with axis 1 of `routing_weights` (the matching feature axis). The
    # survivors are: (nothing from task_vector) and axis-0 of routing_weights
    # (size E=3) → resulting logits shape {3}.
    logits = Nx.dot(task_vector, [0], routing_weights, [1])
    
    # Stable Softmax projects logits to a normalized probability distribution (Lesson 3 & 6)
    stable_softmax(logits)
  end

  # Custom implementation of numerical-stable softmax to prevent logit collapse/overflow.
  # NOTE: this is intentionally duplicated from MoERouter in lesson 12 so that
  # this file can be run with a plain `elixir 14_mini_trinity.exs` without
  # cross-file imports. In a production codebase the right move is to extract
  # this and similar numeric primitives into a shared `MyApp.Nx.Numerics`
  # module that both lesson 12 and the capstone import.
  defn stable_softmax(t) do
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    t_shifted = Nx.subtract(t, max_vals)
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    Nx.divide(exps, sum_exps)
  end
end

defmodule MiniTrinity.Experts do
  @moduledoc """
  Mock experts representing localized expert systems or distinct model APIs (Lesson 6).
  Each expert has different costs and capabilities.
  """

  @doc """
  Dispatches execution to the specified expert.

  - Expert 0: Specialized Math Solver (high cost, succeeds on Math tasks, fails creative tasks).
  - Expert 1: Specialized Creative Writer (high cost, succeeds on Creative tasks, fails math tasks).
  - Expert 2: Cheap Generalist Helper (low cost, has high failure rate on both tasks).
  """
  def execute(0, :math), do: {:ok, "RESULT: x = 2 (Correct Math Solution)", 0.05}
  def execute(0, :creative), do: {:error, "RESULT: Error 400 - Malformed Formula", 0.05}

  def execute(1, :creative), do: {:ok, "RESULT: Once upon a time in Elixir... (Creative Essay)", 0.08}
  def execute(1, :math), do: {:error, "RESULT: Math is too rigid for my soul.", 0.08}

  def execute(2, _task_type) do
    # For testing coordinate warping, we simulate a failure on the first attempt
    {:error, "RESULT: Out of Memory / Timeout", 0.01}
  end
end

defmodule MiniTrinity.Verifier do
  @moduledoc """
  The control loop sensor (Lesson 5 / Control Theory).
  Validates output semantics against target intent without requiring gradients.
  """

  @doc """
  Evaluates the expert's output against the target task type.
  Determines whether to ACCEPT the execution or REVISE (triggering a warp).
  """
  def verify(output, task_type) do
    cond do
      String.contains?(output, "Error") or String.contains?(output, "Out of Memory") ->
        {:revise, "Execution crashed with raw system error."}

      task_type == :math and not String.contains?(output, "Math") and not String.contains?(output, "x =") ->
        {:revise, "Output contains no mathematical formulation."}

      task_type == :creative and not String.contains?(output, "Creative") and not String.contains?(output, "Once") ->
        {:revise, "Output lacks creative narrative style."}

      true ->
        :accept
    end
  end
end

defmodule MiniTrinity.Coordinator do
  @moduledoc """
  The outer governed runtime controller.
  Orchestrates the workflow: Ingest -> Route -> Execute -> Verify -> Replay.
  """

  @doc """
  Ingests a task, initializes its semantic position vector, and runs the execution loop.
  """
  def run(task_name, task_type, task_vector, routing_weights) do
    IO.puts("\n" <> String.duplicate("-", 75))
    IO.puts("INGESTED INTENT: '#{task_name}' (Type: #{inspect(task_type)})")
    IO.puts("TASK COORDINATE: #{inspect(Nx.to_flat_list(task_vector))}")
    IO.puts(String.duplicate("-", 75))

    # Start the stateful execution loop at Turn 1, tracking budget and traces
    loop(task_type, task_vector, routing_weights, _turn = 1, _cost_acc = 0.0, _trace = [])
  end

  # Terminal boundary: Escalated through all turns without success
  defp loop(_task_type, _task_vector, _routing_weights, turn, cost_acc, trace) when turn > 3 do
    IO.puts("\n>> WORKFLOW EXHAUSTED: Max Turns Reached without Verification. Quarantine Action. <<")
    emit_final_trace(:quarantined, cost_acc, trace)
  end

  # Primary execution loop step (Up to 3 turns)
  defp loop(task_type, task_vector, routing_weights, turn, cost_acc, trace) do
    IO.puts("\n--- TURN #{turn} ---")

    # Step 1: Query the Router Head (JIT compiled)
    probs = MiniTrinity.Router.select_expert(task_vector, routing_weights)
    prob_list = Nx.to_flat_list(probs)
    
    # Select the highest-scoring expert
    selected_expert = Nx.argmax(probs) |> Nx.to_number()

    IO.puts("Router Probabilities: [Exp0: #{percent(Enum.at(prob_list, 0))}, Exp1: #{percent(Enum.at(prob_list, 1))}, Exp2: #{percent(Enum.at(prob_list, 2))}]")
    IO.puts("Router Selected: Expert #{selected_expert}")

    # Step 2: Dispatch the task to the selected expert
    IO.puts("Executing Expert #{selected_expert}...")
    {_status, output, cost} = MiniTrinity.Experts.execute(selected_expert, task_type)
    new_cost_acc = cost_acc + cost

    IO.puts("Expert Output: \"#{output}\" (Cost: $#{cost})")

    # Step 3: Pass Output to the Closed-Loop Verifier
    IO.puts("Verifying Output...")
    
    case MiniTrinity.Verifier.verify(output, task_type) do
      :accept ->
        IO.puts(">> VERIFIER DECISION: ACCEPT (Workflow Complete) <<")
        
        emit_final_trace(
          :success, 
          new_cost_acc, 
          trace ++ [%{turn: turn, expert: selected_expert, status: :success, output: output, cost: cost}]
        )

      {:revise, reason} ->
        IO.puts(">> VERIFIER DECISION: REVISE (Reason: #{reason}) <<")
        
        # Shift representation coordinates away from the failed expert's manifold space.
        # This acts as an error-correction signal (Lesson 3 attention-routing / Lesson 5 control theory).
        warped_vector = warp_context_coordinates(task_vector, selected_expert)

        new_step_trace = %{
          turn: turn,
          expert: selected_expert,
          status: :failure,
          output: output,
          cost: cost,
          verifier_feedback: reason
        }

        # Loop again with warped coordinate space (Turn incremented)
        loop(task_type, warped_vector, routing_weights, turn + 1, new_cost_acc, trace ++ [new_step_trace])
    end
  end

  # Shifts the context coordinates representing a "repulsion force" from failed experts
  defp warp_context_coordinates(vector, failed_expert) do
    case failed_expert do
      0 -> 
        IO.puts("Warping coordinate space: Shifting AWAY from Math quadrant (moving left)...")
        Nx.subtract(vector, Nx.tensor([2.5, 0.0])) # Move left along X axis
      1 -> 
        IO.puts("Warping coordinate space: Shifting AWAY from Creative quadrant (moving down)...")
        Nx.subtract(vector, Nx.tensor([0.0, 2.5])) # Move down along Y axis
      2 ->
        IO.puts("Warping coordinate space: Shifting AWAY from Cheap Generalist (escalating to Creative)...")
        # NOTE: this is a HARD RESET to the creative quadrant rather than a
        # relative shift like experts 0 and 1 above. The semantic intent is
        # "expert 2 is our cheap fallback; if it fails, jump straight to the
        # creative specialist's region of the manifold so the router on the
        # next turn picks expert 1 with high confidence". The exact coords
        # [-0.5, 2.0] were chosen so that
        #   logits = [-0.5*2.0 + 2.0*(-1.0), -0.5*(-1.0) + 2.0*2.0, -0.5*1.2 + 2.0*1.2]
        #          = [-3.0, 4.5, 1.8]
        # gives expert 1 a comfortable margin (see manual trace beside Test
        # Case 2 below for the full hand-calculation).
        Nx.tensor([-0.5, 2.0])
    end
  end

  defp percent(val), do: "#{Float.round(val * 100, 1)}%"

  # Output replayable trace
  defp emit_final_trace(status, total_cost, trace) do
    IO.puts("\n" <> String.duplicate("=", 75))
    IO.puts("FINAL REPLAYABLE EXECUTION RECEIPT")
    IO.puts(String.duplicate("=", 75))
    IO.puts("Status:     #{inspect(status)}")
    IO.puts("Total Cost: $#{Float.round(total_cost, 4)}")
    IO.puts("Trace Steps:")
    
    Enum.each(trace, fn step -> 
      IO.puts("  - Turn #{step.turn}: Expert #{step.expert} -> Status: #{inspect(step.status)} (Cost: $#{step.cost})")
      if Map.has_key?(step, :verifier_feedback) do
        IO.puts("      Feedback: #{step.verifier_feedback}")
      end
    end)
    IO.puts(String.duplicate("=", 75) <> "\n")
  end
end

# ===========================================================================
# --- RUNNING THE INTEGRATED CAPSTONE TEST SUITE ---
# ===========================================================================

# Define the specialized routing weights we designed in Lesson 6:
# - Expert 0 (Math Solver) maps heavily to the positive X quadrant
# - Expert 1 (Creative Writer) maps heavily to the positive Y quadrant
# - Expert 2 (Cheap Generalist) is centered at neutral (high weights)
routing_weights = Nx.tensor([
  [ 2.0, -1.0],  # Expert 0 weights
  [-1.0,  2.0],  # Expert 1 weights
  [ 1.2,  1.2]   # Expert 2 weights (Cheap Generalist baseline)
])

# Test Case 1: Standard routing with instant verification (Perfect Match)
# A math task starts in the positive X quadrant -> routed straight to Expert 0.
MiniTrinity.Coordinator.run(
  "Solve x^2 - 4 = 0",
  :math,
  Nx.tensor([2.0, 0.1]),
  routing_weights
)

# Test Case 2: Multi-step routing with failure, coordinate warping, and escalation.
# A creative task starts at the neutral origin -> routed to Expert 2 ->
# Expert 2 crashes/fails -> Verifier triggers warp -> Routed to Expert 1 -> Success!
#
# MANUAL TRACE (hand-computed so the math is reproducible without running the script):
#
#   TURN 1
#     task_vector = [0.1, 0.15]
#     routing_weights rows (R_e = [w_x, w_y]):
#       R_0 = [ 2.0, -1.0]
#       R_1 = [-1.0,  2.0]
#       R_2 = [ 1.2,  1.2]
#     logits = [task · R_0, task · R_1, task · R_2]
#            = [0.1*2.0 + 0.15*(-1.0), 0.1*(-1.0) + 0.15*2.0, 0.1*1.2 + 0.15*1.2]
#            = [0.05, 0.20, 0.30]
#     argmax → Expert 2 (loss-leader pick because the task vector is near origin).
#     Expert 2 returns {:error, "Out of Memory / Timeout", 0.01}.
#     Verifier triggers REVISE → warp_context_coordinates(_, 2) hard-resets
#     task_vector to [-0.5, 2.0].
#
#   TURN 2
#     task_vector = [-0.5, 2.0]
#     logits = [(-0.5)*2.0 + 2.0*(-1.0), (-0.5)*(-1.0) + 2.0*2.0, (-0.5)*1.2 + 2.0*1.2]
#            = [-3.0, 4.5, 1.8]
#     argmax → Expert 1 (creative specialist), as designed.
#     Expert 1 returns {:ok, "Once upon a time in Elixir...", 0.08} → Verifier ACCEPTs.
#
#   FINAL: success in 2 turns, total cost = $0.01 + $0.08 = $0.09.
MiniTrinity.Coordinator.run(
  "Write a prose essay on Erlang actor systems",
  :creative,
  Nx.tensor([0.1, 0.15]),
  routing_weights
)
