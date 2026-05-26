# ===========================================================================
# MINI-LESSON 2: The Computational Engine (Interpretation vs. JIT Compilation)
# ===========================================================================
# In Step 2, our Elixir code was still directing the CPU on how to walk through
# the memory. To achieve hardware-level optimization, we must compile our
# calculations directly to machine code using 'defn' and the EXLA compiler.

# Dynamically pull in Nx and EXLA dependency - Updated to latest Hex versions
Mix.install([
  {:nx, "~> 0.12.0"},
  {:exla, "~> 0.12.0"}
])

# Let's define a module that executes operations inside a numerical definition (defn)
defmodule CompilerDemo do
  import Nx.Defn

  # 'defn' stands for "Numerical Definition".
  # Code inside 'defn' is NOT normal Elixir code. 
  # When Elixir loads this, Nx parses the math, constructs an Abstract Syntax 
  # Tree (AST), and prepares it to be compiled into native assembly for your CPU/GPU.
  defn compute_square_add(a, b) do
    squared = Nx.pow(a, 2)
    Nx.add(squared, b)
  end
end

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("STEP 3: INTERPRETED ELIXIR VS. NATIVE XLA MACHINE CODE")
IO.puts(String.duplicate("=", 75))

# Use large 100_000-element tensors so JIT compile overhead becomes clearly
# distinguishable from steady-state execution time (tiny 3-element tensors are
# dominated by call/dispatch overhead, hiding the actual speedup).
size = 100_000
a = Nx.iota({size}) |> Nx.as_type({:f, 32})
b = Nx.iota({size}) |> Nx.as_type({:f, 32}) |> Nx.multiply(10.0)

# ---------------------------------------------------------------------------
# DEMO 1: Running on the Interpreted Binary Backend
# ---------------------------------------------------------------------------
Nx.global_default_backend(Nx.BinaryBackend)
IO.puts("1. Running with Interpreted Binary Backend (Pure Elixir):")

{time_binary, res_binary} = :timer.tc(fn -> CompilerDemo.compute_square_add(a, b) end)
IO.puts("   - Execution Time: #{time_binary} microseconds")
IO.puts("   - Result[0..2]:   #{inspect(Nx.to_flat_list(Nx.slice(res_binary, [0], [3])))} ... (#{Nx.size(res_binary)} elements total)")
IO.puts(String.duplicate("-", 75))

# ---------------------------------------------------------------------------
# DEMO 2: Running with EXLA Backend (JIT Compilation)
# ---------------------------------------------------------------------------
# Now, let's switch to the EXLA Backend. This backend hands the Nx AST to
# Google's XLA (Accelerated Linear Algebra) compiler to generate optimized binary.
Nx.global_default_backend(EXLA.Backend)
IO.puts("2. Running with EXLA Compiler (First Run - JIT Compilation):")

# The first call will trigger the JIT compiler. XLA will compile our 
# 'compute_square_add' function for the specific CPU instruction set and tensor shapes.
{time_compile, res_compile} = :timer.tc(fn -> CompilerDemo.compute_square_add(a, b) end)
IO.puts("   - Execution Time: #{time_compile} microseconds (includes compiling code to machine instructions!)")
IO.puts("   - Result[0..2]:   #{inspect(Nx.to_flat_list(Nx.slice(res_compile, [0], [3])))} ... (#{Nx.size(res_compile)} elements total)")
IO.puts(String.duplicate("-", 75))

# ---------------------------------------------------------------------------
# DEMO 3: Running with EXLA Backend (Pre-compiled execution)
# ---------------------------------------------------------------------------
IO.puts("3. Running with EXLA Compiler (Second Run - Compiled Direct Execution):")

# Since the code has already been compiled into a machine binary, subsequent
# runs skip compilation entirely and execute native machine code directly at full hardware speed!
{time_compiled, res_exla} = :timer.tc(fn -> CompilerDemo.compute_square_add(a, b) end)
IO.puts("   - Execution Time: #{time_compiled} microseconds")
IO.puts("   - Result[0..2]:   #{inspect(Nx.to_flat_list(Nx.slice(res_exla, [0], [3])))} ... (#{Nx.size(res_exla)} elements total)")
IO.puts(String.duplicate("-", 75))

IO.puts("""
WHAT DID WE JUST OBSERVE?

1. THE INTERPRETED BACKEND (Nx.BinaryBackend):
   This executes inside the standard Elixir VM. It is highly flexible but
   is slow for intensive ML models because it runs as general bytecode.

2. THE JIT COMPILATION STEP (EXLA Backend - First Run):
   Notice how the first run of EXLA is significantly slower. That is because 
   the compiler is building a specialized, high-performance binary for your CPU.

3. THE NATIVE RUNTIME (EXLA Backend - Second Run):
   The second run is lightning fast! The Erlang VM bypassed itself entirely and 
   called the native pre-compiled C++ assembly. This is how Elixir achieves the
   same hardware-level execution speed as PyTorch or TensorFlow (both use XLA to compile too!).

Proceed to run '04_dot_product.exs' to learn the core mathematical operation!
""")
