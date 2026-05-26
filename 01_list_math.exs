# ===========================================================================
# MINI-LESSON 1.1: Standard Elixir List Arithmetic & Pointer Chasing
# ===========================================================================
# In traditional software engineering, we store collections of numbers in
# linked lists or arrays. Let's see how standard Elixir handles this and
# why it represents a performance bottleneck for Machine Learning.

IO.puts("\n" <> String.duplicate("=", 75))
IO.puts("STEP 1: MULTIPLYING 1,000,000 NUMBERS USING STANDARD ELIXIR LISTS")
IO.puts(String.duplicate("=", 75))
IO.puts("""
We are creating two large lists of 1,000,000 floats:
  - list_a = [1.0, 2.0, 3.0, ... 1000000.0]
  - list_b = [1.0, 2.0, 3.0, ... 1000000.0]

We will multiply them element-by-element using standard Enum.zip_with/3.
""")

size = 1_000_000
IO.write("Allocating lists in RAM... ")
# Cast to floats so the benchmark compares float math to float tensor math
# (otherwise we would be comparing integer multiplication to float multiplication)
list_a = Enum.map(1..size, &(&1 * 1.0))
list_b = Enum.map(1..size, &(&1 * 1.0))
IO.puts("Done!")

IO.write("Running list multiplication on CPU... ")
{time, result} = :timer.tc(fn ->
  # Multiply element-by-element. We immediately return the resulting list
  # so the timed block times the full materialization of the products.
  Enum.zip_with(list_a, list_b, fn a, b -> a * b end)
end)
result_sum = Enum.sum(result)
IO.puts("Done! (sum-of-products = #{result_sum})")
IO.puts("  (verifying: first product = #{hd(result)}, last = #{List.last(result)})")

IO.puts("\n" <> String.duplicate("-", 75))
IO.puts("RESULT:")
IO.puts("Elixir List multiplication took: #{time / 1000} ms")
IO.puts(String.duplicate("-", 75))

IO.puts("""
WHY IS THIS TOO SLOW FOR MACHINE LEARNING?

1. THE LINKED LIST PRIMITIVE:
   In Elixir (and the Erlang BEAM virtual machine), a List is represented as a
   singly-linked list. Each item is stored in a separate, disjoint location in
   RAM, pointing to the next node.

   Approximate bytes touched per element on a 64-bit BEAM:
     - list_a: ~16 bytes cons-cell overhead (head + tail pointers) + ~16 bytes boxed float payload = ~32 B/elem
     - list_b: another ~32 B/elem
     - result: another ~32 B/elem
   Compared to a Float32 Nx tensor of size N: just 4*N bytes laid out
   contiguously. For N = 1,000,000 floats the list pipeline can easily
   touch ~96 MB of scattered memory (cons-cells + boxed float overhead),
   vs. ~4 MB contiguous for the tensor path. That's why the cache misses
   we describe below are not a small tax — they dominate runtime.

2. POINTER CHASING OVERHEAD:
   To perform list_a * list_b, the CPU cannot load all data at once. It must:
     a. Lookup the memory address of the first item in list_a and list_b.
     b. Jump to those RAM locations (possibly causing cache misses).
     c. Fetch the floats into registers.
     d. Perform the CPU multiplication instruction.
     e. Find the pointer to the next items and repeat 1,000,000 times!

This constant jump across memory addresses (pointer-chasing) wastes massive
CPU cycles. To do modern AI, we must use Tensors instead.

Proceed to run '02_tensor_math.exs' to see the difference!
""")
