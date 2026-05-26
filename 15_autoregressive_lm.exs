# ===========================================================================
# LESSON 3c: Autoregressive Language Model Generation & Loop
# ===========================================================================
# This script bridges the gap between basic self-attention and a working
# language model (LLM). It demonstrates the entire preprocessing, shape flow,
# causal masking, logit vocabulary projection, and next-token prediction
# loop that forms the heart of autoregressive LLM inference.
#
# Progression demonstrated:
#   text prompt
#   → tokens
#   → token IDs
#   → embedding lookup (shape: {seq} → {seq, hidden_dim})
#   → add positional embeddings
#   → causal masked self-attention (shape: {seq, seq} attention scores)
#   → final hidden states
#   → vocabulary projection logits (shape: {seq, vocab_size})
#   → select last token's logits (shape: {vocab_size})
#   → softmax probabilities & sample next token
#   → repeat (append to sequence)

Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

Nx.global_default_backend(EXLA.Backend)

defmodule AutoregressiveLM do
  import Nx.Defn

  # Numeric stable softmax
  defn stable_softmax(t) do
    max_vals = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    t_shifted = Nx.subtract(t, max_vals)
    exps = Nx.exp(t_shifted)
    sum_exps = Nx.sum(exps, axes: [-1], keep_axes: true)
    Nx.divide(exps, sum_exps)
  end

  # Cross entropy loss for next-token prediction
  # logits: {seq, vocab_size}, targets: {seq}
  defn compute_loss(logits, targets) do
    probs = stable_softmax(logits)
    # Gather the probability of the actual target token for each position
    indices = Nx.stack([Nx.iota({Nx.axis_size(targets, 0)}), targets], axis: 1)
    target_probs = Nx.gather(probs, indices)
    
    # Compute negative log-likelihood (NLL)
    # Avoid log(0.0) with a small epsilon
    loss = Nx.mean(Nx.negate(Nx.log(target_probs + 1.0e-9)))
    loss
  end

  # Compiled Transformer block with causal masking and vocab projection
  # x: {seq, hidden_dim}
  defn forward_pass(x, w_q, w_k, w_v, head_dim, w_vocab) do
    # 1. Project into Q, K, V
    queries = Nx.dot(x, [1], w_q, [0])   # {seq, head_dim}
    keys    = Nx.dot(x, [1], w_k, [0])   # {seq, head_dim}
    values  = Nx.dot(x, [1], w_v, [0])   # {seq, head_dim}

    # 2. Matchmaking (Q · K^T)
    raw_scores = Nx.dot(queries, [1], keys, [1])   # {seq, seq}

    # 3. Apply Causal Mask
    # We want mask[i, j] = 0 if j <= i, and -1.0e9 if j > i
    seq_len = Nx.axis_size(raw_scores, 0)
    row_indices = Nx.iota({seq_len, 1})
    col_indices = Nx.iota({1, seq_len})
    
    # j > i becomes 1, else 0
    mask_condition = Nx.greater(col_indices, row_indices)
    # Multiply by large negative value to simulate -infinity
    causal_mask = Nx.select(mask_condition, -1.0e9, 0.0)

    # Add mask to raw scores before softmax
    masked_scores = Nx.add(raw_scores, causal_mask)

    # 4. Scale and Softmax
    scale_factor = Nx.sqrt(head_dim)
    scaled_scores = Nx.divide(masked_scores, scale_factor)
    attention_weights = stable_softmax(scaled_scores) # {seq, seq}

    # 5. Value mixing
    attention_output = Nx.dot(attention_weights, [1], values, [0]) # {seq, head_dim}

    # 6. Vocab projection to logits
    logits = Nx.dot(attention_output, [1], w_vocab, [0]) # {seq, vocab_size}

    {logits, attention_weights}
  end

  # Sample next token using logits
  # logits: {vocab_size}
  defn sample_token(logits, temperature) do
    # Scale by temperature
    scaled_logits = Nx.divide(logits, temperature)
    probs = stable_softmax(scaled_logits)
    
    # Return both the probabilities and the argmax (greedy token)
    {probs, Nx.argmax(probs)}
  end
end

# --- EXPERIMENT CONFIGURATION ---

# 1. Define our Toy Vocabulary (20 words)
vocab = [
  "<pad>", "The", "cat", "sat", "on", "mat", "dog", "bites",
  "man", "a", "is", "happy", "furry", "friendly", "brown",
  "and", "the", "with", "sleeping", "<eos>"
]

# Quick mapping maps
token_to_id = Enum.with_index(vocab) |> Map.new()
id_to_token = Enum.with_index(vocab) |> Enum.into(%{}, fn {word, idx} -> {idx, word} end)

_vocab_size = length(vocab)
hidden_dim = 8 # Features per vector
_seq_len = 3 # Start sequence: "The cat sat"

# 2. Embedding Table (learned matrix representation of vocab)
# Shape: {vocab_size, hidden_dim} = {20, 8}
# We initialize it with deterministically spread values
embedding_table = Nx.tensor([
  [ 0.1,  0.0, -0.1,  0.2,  0.3,  0.0,  0.1, -0.2], # <pad>
  [ 0.9,  0.2, -0.4,  0.1, -0.1,  0.8,  0.3,  0.5], # The
  [ 0.2,  0.8,  0.6, -0.2,  0.4,  0.1,  0.9, -0.1], # cat
  [-0.1,  0.1,  0.9,  0.7, -0.3,  0.2,  0.1,  0.8], # sat
  [ 0.0,  0.3,  0.1, -0.1,  0.8,  0.6, -0.2,  0.4], # on
  [ 0.3,  0.4, -0.2,  0.9,  0.1,  0.3,  0.6,  0.0], # mat
  [ 0.25, 0.75, 0.5, -0.3,  0.3,  0.15, 0.85,-0.15],# dog
  [-0.15, 0.2,  0.85, 0.65,-0.25, 0.1,  0.05, 0.75],# bites
  [ 0.8, -0.1,  0.3,  0.4,  0.5, -0.2,  0.1,  0.3], # man
  [ 0.45,-0.35, 0.1,  0.2, -0.15, 0.4, -0.3,  0.2], # a
  [-0.2,  0.5,  0.1,  0.3,  0.1,  0.6, -0.4,  0.1], # is
  [ 0.7,  0.6,  0.8, -0.1,  0.2,  0.5,  0.3, -0.2], # happy
  [ 0.15, 0.9,  0.4, -0.1,  0.5,  0.2,  0.7, -0.3], # furry
  [ 0.65, 0.55, 0.45,-0.2,  0.1,  0.35, 0.25,-0.1], # friendly
  [ 0.3,  0.3, -0.4,  0.8,  0.2,  0.1,  0.4,  0.0], # brown
  [-0.3, -0.2,  0.1,  0.0,  0.5,  0.4,  0.6,  0.2], # and
  [ 0.85, 0.15,-0.45, 0.05,-0.15, 0.75, 0.25, 0.45],# the
  [ 0.1,  0.2,  0.3,  0.4, -0.4, -0.3, -0.2, -0.1], # with
  [-0.1, -0.1,  0.5,  0.5,  0.2,  0.3,  0.1,  0.4], # sleeping
  [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  # <eos>
])

# 3. Positional Embedding Table (Learned absolute positions)
# Shape: {max_seq_len, hidden_dim} = {10, 8}
position_embeddings = Nx.tensor([
  [ 0.0,  0.1,  0.2,  0.0,  0.1,  0.2,  0.0,  0.1], # Pos 0
  [ 0.1,  0.2,  0.0,  0.1,  0.2,  0.0,  0.1,  0.2], # Pos 1
  [ 0.2,  0.0,  0.1,  0.2,  0.0,  0.1,  0.2,  0.0], # Pos 2
  [ 0.05, 0.15, 0.25, 0.05, 0.15, 0.25, 0.05, 0.15], # Pos 3
  [ 0.15, 0.25, 0.05, 0.15, 0.25, 0.05, 0.15, 0.25], # Pos 4
  [ 0.25, 0.05, 0.15, 0.25, 0.05, 0.15, 0.25, 0.05], # Pos 5
  [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # Pos 6
  [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # Pos 7
  [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # Pos 8
  [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]  # Pos 9
])

# 4. Attention projection weights and output vocab weights
w_q = Nx.broadcast(0.5, {hidden_dim, hidden_dim})
w_k = Nx.broadcast(0.3, {hidden_dim, hidden_dim})
w_v = Nx.broadcast(0.8, {hidden_dim, hidden_dim})

# Output projection mapping hidden_dim -> vocab_size
# Shape: {hidden_dim, vocab_size} = {8, 20}
w_vocab = Nx.tensor([
  [ 0.1,  0.8,  0.3, -0.1,  0.4,  0.2,  0.5,  0.1, -0.2,  0.3,  0.1,  0.6,  0.2,  0.3,  0.4, -0.3,  0.8,  0.1, -0.1, -0.9],
  [ 0.0,  0.2,  0.9,  0.6, -0.1,  0.4,  0.7,  0.2,  0.1, -0.2,  0.3,  0.1,  0.8,  0.5,  0.2,  0.1,  0.2,  0.3,  0.4, -0.8],
  [-0.2,  0.1,  0.5,  0.8,  0.9, -0.3,  0.1,  0.7,  0.4,  0.1,  0.2,  0.3,  0.5,  0.2,  0.1, -0.1,  0.1,  0.0,  0.2, -0.7],
  [ 0.3, -0.1,  0.2,  0.1,  0.8,  0.7, -0.2,  0.4,  0.6,  0.2,  0.1,  0.5,  0.3,  0.1,  0.9,  0.2, -0.1,  0.4, -0.3, -0.6],
  [ 0.1,  0.4, -0.3,  0.5,  0.2,  0.9,  0.1,  0.3,  0.2,  0.8,  0.4,  0.1,  0.2,  0.6,  0.1,  0.5,  0.4, -0.2,  0.1, -0.5],
  [ 0.2,  0.3,  0.1, -0.2,  0.6,  0.5,  0.8,  0.9, -0.1,  0.3,  0.7,  0.2,  0.1,  0.4,  0.3,  0.1,  0.3,  0.2,  0.6, -0.4],
  [-0.1,  0.9,  0.2,  0.4, -0.3,  0.1,  0.9,  0.8,  0.5,  0.2,  0.1,  0.7,  0.4,  0.1,  0.2,  0.3,  0.9,  0.1, -0.2, -0.3],
  [ 0.0,  0.1,  0.4,  0.7,  0.2,  0.3,  0.1,  0.5,  0.9,  0.6, -0.2,  0.3,  0.6,  0.8,  0.1,  0.4,  0.1,  0.2,  0.5, -0.2]
])

# --- EXECUTION: STEP-BY-STEP GENERATION LOOP ---

prompt = ["The", "cat", "sat"]
token_ids = Enum.map(prompt, fn word -> Map.fetch!(token_to_id, word) end)

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("LESSON 3c: AUTOREGRESSIVE GENERATION SHAPE FLOW AND INFERENCE LOOP")
IO.puts(String.duplicate("=", 75))
IO.puts("Initial Prompt: \"#{Enum.join(prompt, " ")}\"")
IO.puts("Discrete Token IDs: #{inspect(token_ids)}\n")

# Run autoregressive generation loop for 3 steps
final_sequence = Enum.reduce(1..3, token_ids, fn step, current_ids ->
  seq_len = length(current_ids)
  
  # Step A: Convert discrete token IDs to tensor
  ids_tensor = Nx.tensor(current_ids) # shape: {seq_len}
  
  # Step B: Embedding Table Lookup
  # We construct input representations by gathering vectors from our embedding table
  token_embeds = Nx.take(embedding_table, ids_tensor) # shape: {seq_len, hidden_dim}
  
  # Step C: Add Positional Embeddings
  # Grab first seq_len rows from position embeddings table
  pos_embeds = Nx.slice(position_embeddings, [0, 0], [seq_len, hidden_dim]) # shape: {seq_len, hidden_dim}
  x = Nx.add(token_embeds, pos_embeds) # shape: {seq_len, hidden_dim}
  
  # Step D: Causal Attention Forward Pass
  {logits, weights} = AutoregressiveLM.forward_pass(x, w_q, w_k, w_v, Nx.tensor(hidden_dim * 1.0), w_vocab)
  
  # Step E: Predict and Sample Next Token
  # In autoregressive generation, we only look at the logits of the LAST token
  # because causal masking ensures earlier logits cannot see future context.
  last_token_logits = logits[seq_len - 1] # shape: {vocab_size}
  
  # Sample with Temperature = 1.0 (Greedy argmax in our compiled function for stability)
  {_probs, next_id} = AutoregressiveLM.sample_token(last_token_logits, Nx.tensor(1.0))
  next_id_scalar = Nx.to_number(next_id)
  next_word = Map.fetch!(id_to_token, next_id_scalar)
  
  IO.puts("--- Autoregressive Step #{step} ---")
  IO.puts("Current Input Sequence Tokens: #{inspect(Enum.map(current_ids, &Map.fetch!(id_to_token, &1)))}")
  IO.puts("Shape Flow:")
  IO.puts("  1. Token IDs Tensor Shape:          #{inspect(Nx.shape(ids_tensor))}")
  IO.puts("  2. Embedding Lookup Shape:         #{inspect(Nx.shape(token_embeds))}")
  IO.puts("  3. Added Positional Embeds Shape:   #{inspect(Nx.shape(x))}")
  IO.puts("  4. Attention Logits Output Shape:   #{inspect(Nx.shape(logits))}")
  IO.puts("  5. Selected Last Token Logit Shape: #{inspect(Nx.shape(last_token_logits))}")
  IO.puts("  6. Predicted Next Token ID:         #{next_id_scalar} (\"#{next_word}\")")
  
  if step == 1 do
    IO.puts("\nCausal Attention Weight Matrix ({seq_len, seq_len}) for Step 1:")
    IO.inspect(weights)
    IO.puts("  * Notice that the upper triangle is strictly zeros! Causal masking")
    IO.puts("    ensures token i can never attend to token j if j > i.\n")
  end

  # Append new token ID to sequence
  current_ids ++ [next_id_scalar]
end)

# Output final sentence
generated_words = Enum.map(final_sequence, &Map.fetch!(id_to_token, &1))
IO.puts("\nFinal Autoregressively Generated Text:")
IO.puts("  \"#{Enum.join(generated_words, " ")}\"")

# --- DEMONSTRATE TRAINING OBJECTIVE / LOSS ---
# Let's show how the model computes the cross entropy loss for a training sequence
# Input sequence: "The cat sat" -> target tokens: "cat sat on"
IO.puts("\n" <> String.duplicate("-", 75))
IO.puts("TRAINING DEMONSTRATION: NEXT-TOKEN PREDICTION LOSS")
IO.puts(String.duplicate("-", 75))

train_ids = [1, 2, 3] # "The", "cat", "sat"
target_ids = [2, 3, 4] # "cat", "sat", "on" (shifted by one step)

# Compute embeddings
x_train = Nx.take(embedding_table, Nx.tensor(train_ids))
x_train = Nx.add(x_train, Nx.slice(position_embeddings, [0, 0], [3, hidden_dim]))

# Compute logits
{logits_train, _} = AutoregressiveLM.forward_pass(x_train, w_q, w_k, w_v, Nx.tensor(hidden_dim * 1.0), w_vocab)

# Compute NLL / Cross Entropy Loss
loss = AutoregressiveLM.compute_loss(logits_train, Nx.tensor(target_ids))

IO.puts("Input Tokens:  #{inspect(Enum.map(train_ids, &Map.fetch!(id_to_token, &1)))}")
IO.puts("Target Tokens: #{inspect(Enum.map(target_ids, &Map.fetch!(id_to_token, &1)))}")
IO.puts("Training Loss (Negative Log-Likelihood): #{Nx.to_number(loss) |> Float.round(4)}")
IO.puts("===========================================================================\n")
