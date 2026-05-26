# ===========================================================================
# MINI-LESSON 1.2: The Truth About Tensors (Interpreted vs. Compiled)
# ===========================================================================
# You noticed that running 02_tensor_math.exs was actually SLOWER than lists!
# This is a brilliant observation and a vital teaching moment.
# Let's explore exactly why that happened and how we fix it.

# Dynamically pull in Nx and EXLA dependency - Updated to latest Hex versions
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("STEP 2: THE TRUTH ABOUT TENSORS (INTERPRETED VS. COMPILED)")
IO.puts(String.duplicate("=", 75))
IO.puts("""
Why was our first tensor script slower than lists?
Because of two massive factors:

1. THE MEMORY DEVICE GAP (HOST VS. DEVICE):
   If we allocate tensors under standard Elixir (BinaryBackend), they live in
   host memory. If we switch to EXLA and call an eager operation, the system 
   must copy 1,000,000 floats from host RAM to the EXLA execution buffer, perform
   the math, and copy the results back. This transfer overhead ruins performance.

2. EAGER DISPATCH OVERHEAD:
   Running eager tensor math outside a compiled function does not create a fused
   native instruction set.

To unlock contiguous hardware speed, we MUST:
  - Allocate the tensors directly inside the EXLA backend memory space.
  - Compile the operation using a 'defn' block.
""")

size = 1_000_000

# ---------------------------------------------------------------------------
# TEST 1: Interpreted Tensors (Default Binary Backend)
# ---------------------------------------------------------------------------
Nx.global_default_backend(Nx.BinaryBackend)
IO.write("1. Allocating and multiplying interpreted tensors (BinaryBackend)... ")

# Allocated on Host RAM, explicitly cast to f32 so we are comparing
# float-vs-float against the list benchmark in 01_list_math.exs
# (Nx.iota produces integers by default, which would be an apples-to-oranges
# comparison vs. the float list pipeline).
tensor_a_host = Nx.iota({size}) |> Nx.as_type({:f, 32})
tensor_b_host = Nx.iota({size}) |> Nx.as_type({:f, 32})

{time_interpreted, _} = :timer.tc(fn ->
  Nx.multiply(tensor_a_host, tensor_b_host)
end)
IO.puts("Done!")
IO.puts("   - Time taken: #{time_interpreted / 1000} ms")
IO.puts(String.duplicate("-", 75))


# ---------------------------------------------------------------------------
# DEFINE A COMPILED MATH FUNCTION
# ---------------------------------------------------------------------------
defmodule FastMath do
  import Nx.Defn

  # Fuses operations and compiles directly to a native execution instruction set.
  # NOTE: A common misconception is that the body of `defn` is compiled when
  # this module is LOADED. It is not. `defn` compilation is LAZY: the very
  # first call to `FastMath.multiply/2` with a specific {shape, dtype, backend}
  # combination triggers XLA to emit a specialized binary. Subsequent calls
  # with the same shape/dtype hit the compiled cache. That is why "Test 2"
  # below is slow and "Test 3" is fast.
  defn multiply(a, b) do
    Nx.multiply(a, b)
  end
end


# ---------------------------------------------------------------------------
# TEST 2: Compiled Tensors - First Run (EXLA Backend - JIT Compiling)
# ---------------------------------------------------------------------------
# Set EXLA as default BEFORE allocation so memory is allocated natively on device buffers.
# (Note: tensor_a_host and tensor_b_host above were allocated under BinaryBackend, so
# accessing them after this switch would incur an implicit host->device transfer on first use.
# Below we re-allocate fresh tensors under EXLA — and cast them to f32 so the
# benchmark dtype matches Test 1 — to avoid that hidden cost.)
Nx.global_default_backend(EXLA.Backend)
IO.write("2. Allocating and running Compiled Tensors (EXLA - First Run JIT)... ")

# Allocated directly in EXLA memory, dtype-aligned with Test 1's f32 tensors.
tensor_a_device = Nx.iota({size}) |> Nx.as_type({:f, 32})
tensor_b_device = Nx.iota({size}) |> Nx.as_type({:f, 32})

# The very first time EXLA sees this operation, it compiles the math to native CPU/GPU binary.
{time_jit, _} = :timer.tc(fn ->
  FastMath.multiply(tensor_a_device, tensor_b_device)
end)
IO.puts("Done!")
IO.puts("   - Time taken: #{time_jit / 1000} ms (Includes JIT compilation overhead!)")
IO.puts(String.duplicate("-", 75))


# ---------------------------------------------------------------------------
# TEST 2b: MEASURING THE HOST -> DEVICE TRANSFER COST EXPLICITLY
# ---------------------------------------------------------------------------
# To make the "memory device gap" concrete, we time a host-allocated tensor being
# explicitly transferred to the EXLA device buffer using Nx.backend_transfer/2.
# This isolates the copy cost from the compute cost.
#
# We allocate a DEDICATED f32 BinaryBackend tensor here so the transfer cost
# we measure is for the same dtype as the device tensors above (otherwise we
# would be timing a transfer of integer data and reporting it as if it were
# the float pipeline's overhead).
IO.write("2b. Transferring a 1M-element f32 host tensor to the EXLA device... ")
# Copy an already-allocated device tensor to BinaryBackend, giving a valid host f32 tensor
# to measure transfer cost without needing with_default_backend.
host_float = Nx.backend_copy(tensor_a_device, Nx.BinaryBackend)

{time_transfer, _} = :timer.tc(fn ->
  host_float
  |> Nx.backend_transfer(EXLA.Backend)
end)
IO.puts("Done!")
IO.puts("   - Time taken: #{time_transfer / 1000} ms (pure host->device copy of f32, no math)")
IO.puts(String.duplicate("-", 75))


# ---------------------------------------------------------------------------
# TEST 3: Compiled Tensors - Second Run (EXLA Backend - Compiled Direct Execution)
# ---------------------------------------------------------------------------
IO.write("3. Running Compiled Tensors (EXLA - Second Run - Pre-Compiled)... ")

# Now that the machine binary is cached and data lives natively in device memory,
# it runs at pure hardware speeds on raw contiguous bytes.
{time_compiled, _} = :timer.tc(fn ->
  FastMath.multiply(tensor_a_device, tensor_b_device)
end)
IO.puts("Done!")
IO.puts("   - Time taken: #{time_compiled / 1000} ms (Blistering hardware speed!)")
IO.puts(String.duplicate("-", 75))

IO.puts("""
WHAT DID WE LEARN?

1. INTERPRETED TENSORS ARE SLOW (TEST 1):
   In pure interpreted mode, Elixir's default BinaryBackend took hundreds of milliseconds.
   Without compilation, eager tensors are slower than raw lists!

2. THE TRAP OF HOST-TO-DEVICE COPIES (TEST 2):
   Allocating on the host and copying to the device ruins latency. We must allocate 
   directly inside our target hardware backend (using Nx.global_default_backend).

3. NATIVE HARDWARE ACCELERATION (TEST 3):
   Look at Test 3. Once compiled and properly allocated on the device, the 
   multiplication of 1,000,000 numbers runs at blistering speeds
   (typically sub-millisecond on GPU, single-digit ms on CPU — orders of magnitude faster than Test 1).
   That represents a massive 15x to 150x speedup compared to lists or interpreted tensors!

KEY ML RULE: 
Tensors are only fast when compiled and kept in device memory.

Proceed to run '03_compiler.exs' to see how we compile operations!
""")
