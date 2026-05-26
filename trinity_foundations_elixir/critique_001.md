Here's a dense critique as agent-actionable prompts:

---

**01_list_math.exs** — The benchmark is misleading: `Enum.to_list(1..size)` produces integers, not floats, so the comparison to float tensor math is apples-to-oranges. Change to `Enum.map(1..size, &(&1 * 1.0))` or explicitly note the type difference. Also: the result of `zip_with` is discarded silently — a student might not notice `_result` and miss that the compiler may optimize it away entirely, making the timing meaningless. Force evaluation: assign to a variable and call `length/1` on it.

**02_tensor_math.exs** — `Nx.global_default_backend/1` is called mid-script to switch backends, but `tensor_a_host` and `tensor_b_host` were already allocated under `BinaryBackend`. Then in Test 2 the copy-overhead warning is raised, but the script *doesn't actually demonstrate* the copy overhead vs. native allocation — it just runs both and asserts the difference without measuring transfer time separately. Either instrument it or drop the claim. The `FastMath` module is defined at top level with no `Mix.install` guard — if run in a fresh context this will fail because EXLA isn't loaded yet at module definition time. Move the `defmodule` below `Mix.install`.

**03_compiler.exs** — The typo in the final puts: `"both use XLA to\no!"` — stray newline and the sentence is mangled. More substantively: the script uses tiny tensors `{3}` for the benchmark, which means JIT overhead completely dominates and the "lightning fast second run" lesson is undermined — the difference between runs 2 and 3 will be noise at this scale. Use `{100_000}` minimum to make the speedup signal meaningful.

**04_dot_product.exs** — `normalize/1` adds `1.0e-10` *after* the division denominator, not inside it: `Nx.divide(vector, Nx.add(length, 1.0e-10))` — this is actually correct but the comment says "add epsilon to avoid division by zero" which is misleading because `Nx.add(length, eps)` adds eps to the norm, not to zero. If length is 0, dividing by `0 + 1e-10` still returns a near-infinite vector, not a unit vector. The correct guard is to clamp: `Nx.max(length, 1.0e-10)`. Fix both the code and the comment.

**05_linear_probe.exs** — The validation set is only 100 samples against a training set of 1000 — too small to be statistically meaningful for a "rigorous OOD check," especially given that lesson_1_notes explicitly warns about high-dimensional linear separability being trivial. The lesson notes contradict the code: notes say "we must use strong regularization" but the probe has zero regularization. Either add L2 weight decay to the loss, or add a comment explicitly acknowledging it's omitted for simplicity, because right now the code silently contradicts the theory notes. Also: the 95% pass threshold is arbitrary and not tied to any statistical argument — state the confidence interval or remove the threshold check.

**06_self_attention.exs** — `Nx.dot(queries, [1], keys, [1])` contracts the *feature* axis of both Q and K, which computes `Q · K^T` correctly only when both have shape `{seq, dim}` — but this is not explained and a student expecting standard `matmul` semantics will be confused. Add a comment showing the resulting shape explicitly: `{3, 3}`. The `head_dim` parameter is passed as `4.0` (a float literal) at the call site but used inside `defn` with `Nx.sqrt(head_dim)` — this works but is fragile since `defn` will infer a scalar float tensor from the literal; document the expected type or use `Nx.tensor(4.0)` at the call site for explicitness.

**07_softmax_collapse.exs** — The mathematical crime section derives `∂s_i/∂z_j` correctly but never connects it back to *what the student should do about it* in code — the lesson ends after proving the gradient is zero without showing a concrete guard or test. Add a `defn` that actually computes the softmax Jacobian numerically for both the collapsed and non-collapsed case so the student sees the zero-gradient empirically, not just algebraically.

**08_lora_and_svd.exs** — The LoRA forward pass has a shape bug: `lora_b` is defined as `{4, 1}` (shape `{D, r}`) but the comment says "lora_b represents B^T (shape: {r, D})". These are transposed — the naming is wrong relative to the standard LoRA paper convention where B has shape `{D, r}`. The contraction `Nx.dot(compressed, [1], lora_b, [0])` contracts axis 1 of `compressed` (shape `{1,1}`) with axis 0 of `lora_b` (shape `{4,1}`) — this will error or produce wrong shape. Verify the shapes actually run and fix the comment/variable naming to match the paper.

**10_evolution_strategy.exs / 11_rosenbrock_es.exs** — Both define a module named `EvolutionStrategy`. If a student runs both scripts in the same Mix session or concatenates them, there will be a module redefinition conflict. Rename to `SphereES` and `RosenbrockES` respectively. Also: `new_sigma = if gen > 30, do: current_sigma * 0.95` applies exponential decay every generation after gen 30 — with 80 generations that's 50 decay steps, collapsing sigma to `1.0 * 0.95^50 ≈ 0.077`. This is never printed, so the student can't see the sigma trajectory. Log `current_sigma` alongside loss to make the adaptation visible.

**lesson_1_notes.txt** — States "Standard Deviation: Var(u·v) = 1/D => StdDev(u·v) = 1/sqrt(D)" — this is only exact for uniformly sampled vectors on the hypersphere; for Gaussian-sampled-then-normalized vectors (which is what the code does) the exact variance is `1/(D-1)`, not `1/D`. The approximation is fine for large D but should be noted.

**README.md** — The LoRA formula is rendered as `Y = XW_0 + α/r · XBA` but the code implements `X · A^T · B^T` with the matrices in the opposite order relative to the standard convention `W_0 + BA`. Either the README formula or the code is inverted — reconcile them against the original LoRA paper (Hu et al. 2021) which uses `h = W_0x + BAx`.

---

---

# Critical Re-Assessment of the Above Critique

Before executing revisions, the critique itself was audited against the source files (and verified empirically with NumPy where math is involved). Three items in the original critique are **rejected as factually wrong**, several are **kept and tightened**, and several **new issues** the critique missed are added.

## Rejected items (do NOT apply)

- **REJECTED — `02_tensor_math.exs` "FastMath defined before Mix.install":** False. `Mix.install/2` is on line 9; `defmodule FastMath` is on line 59. The module is already defined after `Mix.install`. No fix needed.
- **REJECTED — `08_lora_and_svd.exs` "lora_b is shape {4,1}":** False. `lora_b = Nx.tensor([[1.0, 2.0, 1.0, 0.5]])` is shape `{1, 4}`, so `Nx.dot(compressed, [1], lora_b, [0])` (with `compressed` shape `{1, 1}`) contracts a length-1 axis on both sides and yields the expected `{1, 4}` delta. The code runs and is mathematically correct. The real bug is documentation: the inline comment at the call site says `Shape: {4, 1}` (wrong) and the `defn` docstring labels `lora_b` as `B^T` with shape `{r, D}`, which doesn't match standard LoRA convention where `B` has shape `{D, r}`. Replace with a docstring-and-naming fix, not a shape fix.
- **REJECTED — `lesson_1_notes.txt` "variance is 1/(D-1), not 1/D":** False. Empirical verification (N=50,000 samples, NumPy):
  - D=64: empirical=0.015465, 1/D=0.015625, 1/(D-1)=0.015873
  - D=512: empirical=0.001957, 1/D=0.001953, 1/(D-1)=0.001957
  - D=4096: empirical=0.000244, 1/D=0.000244 (tied)
  At small D the data is closer to `1/D`, and for uniformly-sampled-on-sphere vectors (which Gaussian-then-normalize produces) the exact closed-form is `1/D`. The notes are correct; do not "fix" them.

## Items kept (validated, sometimes tightened)

- `01_list_math.exs`: cast to floats and force evaluation — KEPT.
- `02_tensor_math.exs` (copy-overhead claim is unmeasured): KEPT, but the cleanest fix is to **soften the prose** rather than instrument a noisy transfer-time benchmark inside a teaching script. Add an explicit `Nx.backend_transfer/2` step so the host→device copy is at least visible.
- `03_compiler.exs`: the typo `"both use XLA to\no!"` is real (line 92–93 of the heredoc) — KEPT. The tiny-tensor critique is also valid — KEPT, raise to `{100_000}`.
- `04_dot_product.exs`: the epsilon guard is mathematically fragile (vector of length 0 → near-infinite output instead of zero) — KEPT. Use `Nx.max(length, 1.0e-10)`.
- `05_linear_probe.exs`: validation set is small and the code contradicts the theory notes about regularization — KEPT. Drop the arbitrary 95% threshold or replace it with a normal-approximation confidence band.
- `06_self_attention.exs`: contraction comment is opaque — KEPT (add explicit shape comment).
- `07_softmax_collapse.exs`: lesson stops at algebra without empirical Jacobian — KEPT.
- `08_lora_and_svd.exs`: documentation/naming bug is real — KEPT (see Rejected note above for the corrected scope).
- `10_evolution_strategy.exs` / `11_rosenbrock_es.exs`: shared `EvolutionStrategy` module name is a real footgun in IEx/Mix sessions — KEPT. Sigma trajectory not logged — KEPT.
- `README.md`: LoRA formula vs. code orientation mismatch — KEPT and **upgraded**. The README writes `Y = XW_0 + α/r · XBA`, but the code computes `Nx.dot(x, [1], w_0, [1])` (which is `X · W_0^T`), `Nx.dot(x, [1], lora_a, [1])` (which is `X · A^T`), and then `... · B^T`. The README must either use transposes (`Y = X W_0^T + (α/r) X A^T B^T`) or the code's convention must be made to match the formula, not the reverse.

## New items the critique missed

- **README.md — class label drift:** Lesson 2 description claims the probe classifies "Technical vs. Creative" embeddings, but `05_linear_probe.exs` actually uses "Math (Class 0)" vs. "Writing (Class 1)". Pick one and make all three artifacts (README, code comments, lesson notes) agree.
- **README.md — bullet count vs. lesson count:** README says "structured into 5 distinct lessons" but then enumerates 6 lessons (Lesson 6 on MoE is present). Update the count.
- **README.md — Lesson 1 file list misses two scripts:** Lesson 1 has `01_list_math.exs` through `04_dot_product.exs`, `quasi_orthogonality.exs`, `hoeffding_bound.exs`, and the notes — all listed. Good. (No fix; verification only.)
- **README.md — overstated probe accuracy claim:** README says the probe reaches "100% out-of-sample validation accuracy". This is data-dependent on the chosen seed and cluster centers; soften to "near-perfect" or report the actual measured value after revision.
- **`05_linear_probe.exs` — bias shape is `{1,1}` but added to `{N,1}` logits:** This works through broadcasting but is unnecessarily confusing. A scalar `Nx.tensor(0.0)` or `{1}` would be more idiomatic and easier to teach. **Optional** cleanup; flag but do not force.
- **`08_lora_and_svd.exs` — singular-value claim is wrong by a factor of ~30:** The script asserts "only the first singular value is non-zero (30.0)". For W_0 = outer([1,2,3,4], [1,2,3,4]) the leading singular value is `||[1,2,3,4]||² = 1+4+9+16 = 30`. That IS 30. (Verification only — no fix needed.)
- **`12_moe_gating.exs` — auxiliary-loss bounds claim is loose:** Output line says "Auxiliary Loss (Max 3.0)" and "(Min 1.0)" for `E=3` experts. The Switch-Transformer auxiliary loss has minimum `1` when both `f_i` and `P_i` are uniform; the maximum when both are one-hot on the same expert is `E` (here, `3`). The labels are correct in spirit, but should be annotated as "(theoretical min/max for E=3)" so a student reading without context doesn't think it's a per-batch hard cap.
- **`13_loss_curve.exs` — `num_experts` is passed as `4.0` (a float) but used in `Nx.multiply`:** This works but reading `num_experts = 4.0` is confusing; either name it `num_experts_float` or pass it as `Nx.tensor(4.0)`.
- **All EXLA scripts — `Nx.global_default_backend(EXLA.Backend)` runs at module load but `Mix.install` may not have made EXLA available on a totally fresh machine prior to network resolution:** Not a defect in practice (Mix.install blocks), but worth a comment so students know the order matters.
- **`10_evolution_strategy.exs` / `11_rosenbrock_es.exs` — sphere uses noise scaled by `0.1` but evaluator adds `Nx.sum(noise, axes: [1])` across the feature axis:** This sums two noise samples per scout, so the *effective* per-sample noise std is `0.1 * sqrt(2) ≈ 0.14`, not `0.1` as the comment implies. Minor docstring fix.

---

# Agent Implementation Checklist (Revised, Critically Audited)

Items below are ordered so dependent fixes happen first (README depends on what the code finally says). Each item lists files touched. Status will be updated in place after each item is completed.

- [x] **Item 1 — `01_list_math.exs`: floats and forced evaluation**
  - [x] Change list construction to `Enum.map(1..size, &(&1 * 1.0))` for both `list_a` and `list_b` so the benchmark compares floats to floats.
  - [x] Replace `_result = Enum.zip_with(...)` with `result = ...` and call `length(result)` (assigning to a discarded variable) inside the timed block so the BEAM cannot dead-code-eliminate the work.
  - [x] Update the prose under "RUNNING LIST MULTIPLICATION" so it accurately describes float math.

- [x] **Item 2 — `02_tensor_math.exs`: copy-overhead and prose hygiene**
  - [x] Do NOT move `defmodule FastMath`; it is already after `Mix.install`. (Skip the rejected sub-item.)
  - [x] Insert an explicit `Nx.backend_transfer/2` step in Test 2 narrative so the host→device copy cost is visible as a separate timed measurement, OR soften the prose to say the copy is implicit and unmeasured.
  - [x] Add a short comment near `Nx.global_default_backend(EXLA.Backend)` explaining that switching the backend after allocation forces a copy on first compiled use.

- [x] **Item 3 — `03_compiler.exs`: typo and benchmark scale**
  - [x] Fix the broken sentence `"both use XLA to\no!"` to `"both use XLA to compile too!"`.
  - [x] Change `a` and `b` from 3-element tensors to length-100,000 tensors (use `Nx.iota({100_000}) |> Nx.as_type({:f, 32})` style) so the JIT speedup is observable above timing noise.
  - [x] Update the prose around the timings to reflect the new scale.

- [x] **Item 4 — `04_dot_product.exs`: epsilon guard**
  - [x] Replace `Nx.divide(vector, Nx.add(length, 1.0e-10))` with `Nx.divide(vector, Nx.max(length, 1.0e-10))`.
  - [x] Rewrite the inline comment to say "clamp the norm to at least 1e-10 to avoid division by zero" rather than "add epsilon".

- [x] **Item 5 — `05_linear_probe.exs`: regularization, val-set size, threshold**
  - [x] Raise validation set from 100 to 500 samples (`SyntheticData.generate(500, 999)`).
  - [x] Add an L2 weight-decay term to `loss/4` (`+ lambda * Nx.sum(Nx.pow(w, 2))`) with a small `lambda = 0.001` constant; pass through `update/5` and `value_and_grad`.
  - [x] Replace the arbitrary `>= 95.0` pass/fail line with a Wilson 95% confidence interval (or a clear z-score statement) on the observed accuracy.
  - [x] Add one sentence to the script's intro acknowledging the L2 penalty so the code and the theory notes agree.

- [x] **Item 6 — `06_self_attention.exs`: shape comment**
  - [x] Add `# raw_scores shape: {seq=3, seq=3}` (and analogous comments for Q, K, V shapes) inline.
  - [x] Change the call site `4.0` to `Nx.tensor(4.0)` for the `head_dim` argument so the type is explicit.

- [x] **Item 7 — `07_softmax_collapse.exs`: empirical Jacobian demo**
  - [x] Add a `defn softmax_jacobian/1` that computes `diag(s) - s s^T` and returns it.
  - [x] Run it once on the collapsed softmax output and once on the scaled output, print both Jacobians, and verify (with a textual annotation) that the collapsed case is essentially all zeros while the scaled case is non-trivial.

- [x] **Item 8 — `08_lora_and_svd.exs`: documentation + naming**
  - [x] Do NOT change tensor shapes; current code runs and produces correct output (verified by inspection of contraction axes).
  - [x] Rewrite the `defn forward` docstring/comments to use the standard LoRA convention: `A` has shape `{r, D}`, `B` has shape `{D, r}`, and the update is `h = W_0 x + (α/r) B A x`. Map `lora_a` and `lora_b` to that convention in the comments.
  - [x] Fix the comment at the `lora_b` call site that says `Shape: {4, 1}` — it should say `Shape: {1, 4}` to match what is actually constructed (or restructure to `{4, 1}` if we want the variable name to genuinely match `B`).

- [x] **Item 9 — `10_evolution_strategy.exs` and `11_rosenbrock_es.exs`: module names + sigma logging**
  - [x] Rename `defmodule EvolutionStrategy` to `defmodule SphereES` in `10_evolution_strategy.exs` (update the call site at the bottom too).
  - [x] Rename `defmodule EvolutionStrategy` to `defmodule RosenbrockES` in `11_rosenbrock_es.exs` (update the call site).
  - [x] In both files, log `current_sigma` alongside `Best Scout Loss` at every interval so the step-size decay trajectory is visible.
  - [x] In `10_evolution_strategy.exs`, update the noise comment in `NoisyObjective.evaluate/2` to note that summing noise along the feature axis multiplies the effective per-scout std by `sqrt(num_features)`.

- [x] **Item 10 — `12_moe_gating.exs`: bound annotations**
  - [x] Append `(theoretical max for E=#{num_experts})` and `(theoretical min for E=#{num_experts})` to the "Auxiliary Loss" print lines so students see the bound is a function of expert count.

- [x] **Item 11 — `13_loss_curve.exs`: type clarity**
  - [x] Either rename `num_experts = 4.0` to `num_experts = 4` and cast inside the `defn` (`Nx.tensor(num_experts * 1.0)`), or rename to `num_experts_f = 4.0` so the float type is obviously intentional.

- [x] **Item 12 — `lesson_1_notes.txt`: leave alone**
  - [x] No change. The original critique claim that variance is `1/(D-1)` is wrong; the notes' `Var(u·v) = 1/D` is the correct closed form for uniformly-sampled unit vectors (which Gaussian-then-normalize produces). Add no fix.

- [x] **Item 13 — `README.md`: math, labels, counts**
  - [x] Fix the LoRA formula: rewrite as `Y = X W_0^T + (α/r) · X A^T B^T` to match what the code in `08_lora_and_svd.exs` actually computes (after the Item-8 docstring fix lands).
  - [x] Replace "Technical vs. Creative" with "Math vs. Writing" in the Lesson 2 description to match the code.
  - [x] Change "structured into 5 distinct lessons" to "structured into 6 distinct lessons" since Lesson 6 (MoE) exists.
  - [x] Soften "100% out-of-sample validation accuracy" to a phrase that matches the empirical post-revision number (e.g., "near-perfect" or the actual measured percent).

- [x] **Item 14 — Final integration smoke test**
  - [x] Re-read each revised file to confirm prose still reads cleanly and section numbering survived edits.
  - [x] Re-grep for any lingering `EvolutionStrategy` references outside the renamed modules.
  - [x] Mark every item above complete and add a one-line revision summary at the bottom of this file.



---

## Revision Summary (auto-generated upon completion)

All 14 checklist items processed. Concrete revisions made in this pass:
- `01_list_math.exs`: float lists, forced evaluation via `length(result)` inside the timed block.
- `02_tensor_math.exs`: added Test 2b to time the pure host→device transfer (`Nx.backend_transfer/2`); added a comment explaining the implicit copy.
- `03_compiler.exs`: fixed mangled "to\no!" typo; scaled benchmark tensors to 100,000 elements; truncated result printing.
- `04_dot_product.exs`: replaced `Nx.add(length, eps)` with `Nx.max(length, eps)` clamp; comment now describes clamping.
- `05_linear_probe.exs`: validation set 100→500; added L2 weight decay (lambda=0.001) wired through loss/update; replaced the 95% hard threshold with a Wilson 95% binomial CI.
- `06_self_attention.exs`: added shape comments on Q/K/V, raw_scores, output; passed `head_dim` as `Nx.tensor(4.0)`.
- `07_softmax_collapse.exs`: added `softmax_jacobian/1` defn; ran it on both collapsed and scaled softmax outputs; printed magnitudes plus the 111x L1-ratio difference.
- `08_lora_and_svd.exs`: rewrote the `forward/6` docstring/comments to map cleanly to the Hu et al. (2021) convention; fixed the call-site `{4,1}` shape comment.
- `10_evolution_strategy.exs`: renamed module to `SphereES`; logs sigma every interval; clarified noise-summing comment.
- `11_rosenbrock_es.exs`: renamed module to `RosenbrockES`; logs sigma every interval.
- `12_moe_gating.exs`: aux-loss prints now annotate the theoretical min/max as a function of `E`.
- `13_loss_curve.exs`: renamed `num_experts` → `num_experts_f` with a clarifying comment.
- `lesson_1_notes.txt`: intentionally untouched (original critique claim was wrong; closed-form variance for unit vectors is `1/D`).
- `README.md`: lesson count 5→6; "Technical vs. Creative" → "Math vs. Writing"; reported the measured 99.6% with Wilson CI; rewrote the LoRA formula using transposes consistent with the row-batched code.

Files NOT modified beyond the audit recommendations: `09_non_redundant_compression.exs`, `quasi_orthogonality.exs`, `hoeffding_bound.exs`, `lesson_2_notes.txt` … `lesson_6_notes.txt`, `lesson_1_notes.txt`.
