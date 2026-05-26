<p align="center">
  <img src="assets/ml_musings.svg" width="200" height="200" alt="ML Musings Logo" />
</p>

<p align="center">
  <a href="https://github.com/nshkrdotcom/ml_musings">
    <img src="https://img.shields.io/github/stars/nshkrdotcom/ml_musings?style=for-the-badge&logo=github&color=38bdf8&logoColor=fff" alt="GitHub Repository" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-purple?style=for-the-badge&color=a855f7" alt="MIT License" />
  </a>
</p>

# TRINITY Foundations: High-Dimensional Geometry, Attention, PEFT, and Black-Box Optimization

This repository houses a comprehensive, hands-on educational curriculum designed to explore the mathematical, geometric, and computational foundations undergirding modern AI models and coordination systems like **TRINITY**.

All implementations are written from scratch in **Numerical Elixir (`Nx`)** and compiled natively to your hardware via **EXLA (Elixir XLA Compiler)**. The scripts are fully configured to run with native **CUDA GPU acceleration** (specifically tested on the **NVIDIA GeForce RTX 5060 Ti**), extracting absolute peak performance from the silicon.

---

## 📚 Curriculum Structure & File Catalog

The curriculum is structured into 6 distinct lessons, progressing from core high-dimensional tensor mechanics through MoE-style routing and black-box optimization.

### 📐 Lesson 1: The Geometry of High-Dimensional Spaces (Quasi-Orthogonality)
This lesson investigates the unique geometric properties of the spaces where LLM vector representations (embeddings and hidden states) reside. It validates the phenomenon of **Quasi-Orthogonality**—often called the "Blessing of Dimensionality"—which explains how models store independent semantic concepts in perpendicular directions.

*   **[01_list_math.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/01_list_math.exs)**: *CPU Linked-List Baseline*. Demonstrates why standard Elixir lists (singly-linked lists with pointer-chasing overhead and lack of memory contiguity) are computationally prohibitive for machine learning.
*   **[02_tensor_math.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/02_tensor_math.exs)**: *Eager Device Tensors*. Benchmarks contiguous, binary-backed tensors on CPU and GPU, highlighting the orders-of-magnitude speedup achieved by hardware acceleration.
*   **[03_compiler.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/03_compiler.exs)**: *JIT Graph Compilation*. Explains the difference between eager Elixir tensor evaluation and JIT-compiled computation graphs (`defn` with the `EXLA` backend) which compile directly into LLVM/PTX assembly.
*   **[04_dot_product.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/04_dot_product.exs)**: *Cosine Similarity Mechanics*. Implements vector L2-normalization and dot-product similarity, showing how vector angles represent semantic alignment.
*   **[quasi_orthogonality.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/quasi_orthogonality.exs)**: *Empirical Geometry Simulation*. Randomly samples pairs of unit vectors across varying dimensions (from $D=2$ to $D=8192$) and plots their similarity distributions using a custom **ASCII 95% Dispersion Bar** to visually witness the contraction of similarity density towards exactly $0.0$ (perpendicularity).
*   **[hoeffding_bound.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/hoeffding_bound.exs)**: *Mathematical Bound Validation*. Compares empirical vector similarity distributions against the theoretical exponential decay bounds calculated via **Hoeffding's Inequality**.
*   **[lesson_1_notes.txt](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/lesson_1_notes.txt)**: *Theoretical Notes*. Mathematical derivations of quasi-orthogonality, measure concentration, and the Curse vs. Blessing of Dimensionality.

---

### 🔎 Lesson 2: Linear Probing and Representational Geometry
Explores how semantic information is represented inside an LLM's activation space, and how we can extract that information using a **Linear Probe**—a simple, non-destructive classifier that learns a separating hyperplane.

*   **[05_linear_probe.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/05_linear_probe.exs)**: *Differentiable Logistic Classifier*. Implements a binary linear probe from scratch. Uses **Automatic Differentiation (`Nx.grad/2`)** and batch gradient descent on the GPU to learn a separating hyperplane that extracts concept vectors (classifying "Math" vs. "Writing" embeddings). The latest run reaches ~99.6% validation accuracy (Wilson 95% CI [98.55%, 99.89%], N=500); see `05_linear_probe.exs` for the exact print.
*   **[lesson_2_notes.txt](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/lesson_2_notes.txt)**: *Conceptual Notes*. Deep dive into embedding lookups, the Manifold Hypothesis in deep learning, coordinate systems, and the geometric meaning of separating hyperplanes.

---

### 🔄 Lesson 3: The Self-Attention Mechanism (Queries, Keys, and Values)
De-magic-ifies the core engine of modern Transformers by implementing a fully functional, differentiable Self-Attention head from scratch, framing it as an **Information Routing System**.

*   **[06_self_attention.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/06_self_attention.exs)**: *Attention Routing Head*. Implements the Query, Key, and Value ($Q, K, V$) projections, the scaled dot-product similarity matrix, stable softmax, and attention routing weights to update token representations dynamically.
*   **[07_softmax_collapse.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/07_softmax_collapse.exs)**: *Softmax Entropy & Numerical Stability*. Illustrates why the division by the scaling factor $\sqrt{D_k}$ is mathematically necessary. Geometrically demonstrates **Softmax Collapse** (vanishing gradients/over-saturation) in high-dimensional attention and evaluates numerical overflow behaviors under extreme logit distributions.
*   **[lesson_3_notes.txt](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/lesson_3_notes.txt)**: *Attention Physics*. Detailed notes on semantic vector shifting, entropy dynamics, and mathematical proofs of softmax scale-invariance.

---

### 📉 Lesson 4: Parameter-Efficient Adaptation (PEFT, Rank, SVD, and LoRA)
Explores how to fine-tune massive pre-trained model weights efficiently. We perform surgery on weight matrices using **Singular Value Decomposition (SVD)** and implement a low-rank bypass adapter (LoRA) from scratch.

*   **[08_lora_and_svd.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/08_lora_and_svd.exs)**: *SVD Surgery & LoRA Layer*. Decomposes a redundant weight matrix on the GPU, truncates it to Rank-1, and implements the parallel forward pass of a **Low-Rank Adaptation (LoRA)** layer. The code uses row-batched inputs (one token per row of $X$), so the implementation expresses the Hu et al. (2021) update $h = W_0 x + (\alpha/r) \, B A x$ as $Y = X W_0^\top + (\alpha/r) \, X A^\top B^\top$.
*   **[09_non_redundant_compression.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/09_non_redundant_compression.exs)**: *Lossy Compression Benchmark*. Generates a full-rank, non-redundant random matrix, collapses it down to Rank-1 using SVD, and calculates exact information loss metrics using **Mean Squared Error (MSE)** and **Frobenius Norm** reconstruction errors on the device.
*   **[lesson_4_notes.txt](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/lesson_4_notes.txt)**: *Comparative Notes*. Compares LoRA's low-rank coordinate bypass with **Singular Value Fine-Tuning (SVF)**. Explains why adapting *only* diagonal singular values ($\Sigma$) achieves zero inference computational overhead.

---

### 🎲 Lesson 5: Black-Box Optimization (The Evolution Strategy)
Explores optimization landscapes where backpropagation is mathematically impossible, such as coordinating and routing between external, isolated APIs (Claude, Gemini, GPT-4) across non-differentiable network boundaries.

*   **[10_evolution_strategy.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/10_evolution_strategy.exs)**: *GPU-Compiled Sphere Optimizer*. Implements an **Evolution Strategy (ES)** from scratch in Elixir. Optimizes a noisy, 2D Sphere landscape $f(x_1, x_2) = x_1^2 + x_2^2 + \text{noise}$ to find the origin $[0.0, 0.0]$ using a population of stochastic "scout" mutations and weighted recombinations, completely bypassing gradients.
*   **[11_rosenbrock_es.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/11_rosenbrock_es.exs)**: *Rosenbrock Curved Valley Solver*. Tests the ES optimizer on the highly non-separable, curved **Rosenbrock function** (the "banana function") to demonstrate how population-level average scout vectors navigate steep, narrow valleys.
*   **[lesson_5_notes.txt](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/lesson_5_notes.txt)**: *API Boundary Coordination Notes*. Documents the three factors that block backpropagation across third-party models (Weight Isolation, Discrete Token Discontinuity, and Network Socket Barriers). Explores the mathematical architecture of **`sep-CMA-ES`** used in TRINITY and why diagonal covariance scaling achieves $O(D)$ linear complexity.

### 🧩 Lesson 6: Mixture of Experts (MoE) & Gating Load Balancing
This lesson addresses efficient model scaling using sparse parallel "expert" neural networks. It covers routing activations and introduces load-balancing techniques to prevent GPU bottlenecks and coordinate API delegation.

*   **[12_moe_gating.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/12_moe_gating.exs)**: *MoE Router & Load Loss*. Implements a Top-1 sparse routing gating projection and compiles an **Auxiliary Load-Balancing Loss** from scratch on the GPU, evaluating collapsed vs. balanced routing distributions.
*   **[13_loss_curve.exs](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/13_loss_curve.exs)**: *Gating Imbalance Penalty Curve*. Simulates multiple token load distributions across 4 parallel experts and calculates simulated auxiliary losses on the GPU, printing a custom ASCII penalty bar chart to visualize the skew penalty.
*   **[lesson_6_notes.txt](file:///home/home/p/g/n/ml_musings/trinity_foundations_elixir/lesson_6_notes.txt)**: *API System Balance Notes*. Compares intra-model FFN block routing with TRINITY's external API coordination, documenting the critical operational, financial, and rate-limiting impacts of Expert Collapse in multi-agent environments.

---

### 🏛️ Lesson 7: Capstone - The Mini-TRINITY Framework
This capstone project compiles all foundational components developed in Lessons 1–6 into a stateful, governed execution loop substrate. It implements a closed-loop control write-path routing and executing intents with runtime coordinate warping for dynamic model escalation.

*   **[14_mini_trinity.exs](file:///home/home/p/g/n/ml_musings/14_mini_trinity.exs)**: *The Mini-TRINITY Substrate*. Integrates JIT-compiled routing projections (`defn` + `stable_softmax`), diverse mock expert models, custom semantic verification sensors, and control-loop representation warping to adjust routing vectors on expert execution failures.

---


## ⚡ Prerequisites & System Installation

Ensure Erlang, Elixir, and the required CUDA drivers are installed on your Linux system.

```bash
# Update Ubuntu package lists and install Elixir + Erlang BEAM VM
sudo apt-get update
sudo apt-get install -y erlang elixir

# Verify Elixir installation
elixir --version
```

### Hex Package Management
All scripts leverage Elixir's runtime package installer (`Mix.install/2`) to pull the absolute latest stable versions of Numerical Elixir and its XLA compiler bindings directly from Hex:
*   `nx ~> 0.12.0`
*   `exla ~> 0.12.0`

---

## 🚀 Execution Guide

Run any of the curriculum scripts directly from your bash terminal.

### Running with Native NVIDIA CUDA GPU Acceleration
To compile computation graphs directly into optimized CUDA GPU kernels on your **NVIDIA GeForce RTX 5060 Ti**, set the `XLA_TARGET` environment variable before executing:

```bash
# Execute Lesson 1 empirical simulations
XLA_TARGET=cuda12 elixir quasi_orthogonality.exs
XLA_TARGET=cuda12 elixir hoeffding_bound.exs

# Run the JIT compiler benchmark
XLA_TARGET=cuda12 elixir 03_compiler.exs

# Train the Linear Probe classifier
XLA_TARGET=cuda12 elixir 05_linear_probe.exs

# Run the Self-Attention Head implementation
XLA_TARGET=cuda12 elixir 06_self_attention.exs

# Execute LoRA low-rank adaptation
XLA_TARGET=cuda12 elixir 08_lora_and_svd.exs

# Run Lesson 5 Evolution Strategy Optimizers
XLA_TARGET=cuda12 elixir 10_evolution_strategy.exs
XLA_TARGET=cuda12 elixir 11_rosenbrock_es.exs

# Run Lesson 6 Mixture of Experts Router & Loss Curve
XLA_TARGET=cuda12 elixir 12_moe_gating.exs
XLA_TARGET=cuda12 elixir 13_loss_curve.exs

# Run Lesson 7 Capstone Mini-TRINITY Framework
XLA_TARGET=cuda12 elixir 14_mini_trinity.exs
```

### Running on CPU (Fallback)
If a CUDA GPU is not available, EXLA will automatically fall back to CPU compilation, or you can run using standard native CPU execution by omitting `XLA_TARGET`:

```bash
elixir 10_evolution_strategy.exs
elixir 13_mini_trinity.exs
```


---

## 🔬 Key Pedagogical Highlights

1.  **Hardware-Level Contiguity**: Understand the transition from Elixir's heap-allocated linked lists to flat, binary-backed hardware buffers, speeding up basic matrix multiplications from **30+ ms** to **0.5 ms**.
2.  **No Arbitrary Placeholders**: Every script is self-contained, using real statistical distributions, empirical bounds checking, and rigorous loss metrics.
3.  **Automatic Differentiation**: See how `Nx.grad` automatically traverses computation graphs, compiling backpropagation steps to parallelized hardware kernels.
4.  **Black-Box Robustness**: Witness how Evolution Strategies filter out heavy evaluation noise ($\sigma = 0.1$ random variance) through population-level mean recombinations, converging reliably where traditional gradient descent would stall.
