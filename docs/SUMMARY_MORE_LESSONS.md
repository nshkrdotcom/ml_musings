### 1. The KV Cache (Key-Value Cache) and the Memory-Bandwidth Bottleneck

During the autoregressive generation (decoding) phase of a Decoder-only Transformer, the model predicts one token at a time. The prediction of token $t$ depends on the representation of all prior tokens $1$ to $t-1$. 

To avoid redundantly recomputing the Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) projections for every historical token at each generation step, the model computes them once and stores the Key and Value tensors in GPU memory. This storage is the **KV Cache**.

```text
       Autoregressive Step t:
       Input Token (t) ──► Projections ──► Q_t
                                           K_t ──┐ 
                                           V_t ──┼─► [Append to Cache]
                                                 │
       Historical KV Cache (Tokens 1..t-1) ──────┘
         Layers:   [ L_1, L_2, ... L_32 ]
         Memory:   [ K_1..t-1, V_1..t-1 ] ──► Loaded from HBM to SRAM every step
```

#### The Memory Footprint Formula
The physical size of the KV Cache (in bytes) for a single generation sequence of length $T$ is given by:

$$|KV| = 2 \cdot L \cdot H_{kv} \cdot d \cdot T \cdot \text{bytes-per-parameter}$$

Where:
*   $L$ is the number of transformer layers.
*   $H_{kv}$ is the number of Key-Value heads (which is smaller than the Query heads in Grouped-Query Attention).
*   $d$ is the head dimension (typically 128).
*   $T$ is the sequence length (context window).
*   The factor of $2$ accounts for storing both Keys and Values.

For Llama-3.1-8B-Instruct ($L = 32$, $H_{kv} = 8$, $d = 128$, FP16 precision = $2$ bytes):
*   At $T = 8,192$ tokens:
    $$|KV| = 2 \times 32 \times 8 \times 128 \times 8,192 \times 2 = 1,073,741,824 \text{ bytes} \approx 1.07\text{ GB}$$
*   At $T = 131,072$ tokens (128K context limit):
    $$|KV| \approx 17.18\text{ GB}$$

#### The Hardware Bottleneck: Compute-Bound vs. Memory-Bound
During the prefill phase (processing the input prompt), the GPU processes all tokens in parallel. This utilizes high matrix-computation density (FLOPs) and is **compute-bound**. 

During the decode phase (generating new tokens one-by-one), the batch size is effectively $1$ token. The GPU must load the entire model weights (16 GB for an 8B model) and the entire historical KV cache from High Bandwidth Memory (HBM) into its on-chip SRAM just to calculate a single token's attention. The arithmetic intensity (ratio of FLOPs to memory bytes transferred) is extremely low. Consequently, decoding is heavily **memory-bandwidth bound**. 

Compressing the KV cache directly reduces the volume of bytes transferred over the memory bus, resolving the HBM bandwidth bottleneck and allowing larger batch sizes.

---

### 2. The Structural Divergence of Key ($K$) vs. Value ($V$) Spaces

Historically, KV cache compression algorithms treated Key and Value tensors symmetrically, applying identical quantization or dimensionality reduction pipelines to both sides. However, Keys and Values serve different algebraic roles in the self-attention equation:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_k}}\right)\mathbf{V}$$

This difference in functionality results in a stark divergence in their geometric and informational structures:

```text
       KEY SPACE (Low Intrinsic Rank)               VALUE SPACE (High Entropy)
       
              h_2 ▲                                         h_2 ▲     ●     ○
                  │    ●   ● (Coordinated                   │  ○     ○    ●
                  │   ●   ●   Subspace)                     │     ●    ○   ●
                  │  ●   ●                                  │   ○    ●   ○
                  └────────────────► h_1                    └────────────────► h_1
           Singular values decay rapidly.             Singular values are flat.
           PCA-compressible.                          Isotropic / uniform on sphere.
```

#### The Key Space ($K$)
Keys are used to compute attention weights—effectively executing a matchmaking operation with the Query. Because natural language and programming syntax follow highly structured, low-dimensional coordinate patterns (such as grammar rules, part-of-speech dependencies, and positional markers), the keys represent a highly coordinated manifold. 

When you run Singular Value Decomposition (SVD) on the accumulated Key cache, the **singular values decay rapidly**. This means the Key space is structurally **low-rank**; a small number of orthogonal directions (e.g., 192 out of 1024) capture over $99.5\%$ of the geometric variance.

#### The Value Space ($V$)
Values represent the actual payload or semantic content retrieved once a match is established. To maximize the expressiveness of the model, this content must be highly diverse and informational. 

When SVD is run on the Value cache, the **singular values are almost flat**. The Value space behaves like an isotropic, high-entropy distribution (effectively uniform across the hypersphere). It lacks low-rank properties, meaning any attempt to compress $V$ using linear projection (PCA/SVD) results in severe information loss and downstream model degradation. Value compression must instead be handled via non-linear vector quantization.

---

### 3. Rotary Position Embedding (RoPE) Obscuration

Llama-type architectures use **Rotary Position Embeddings (RoPE)** to encode relative positional information. RoPE applies a position-dependent rotation matrix to the Query and Key vectors. 

#### The Mathematical Problem
For a vector $\mathbf{x} \in \mathbb{R}^d$ at token position $p$, RoPE splits the vector into $d/2$ two-dimensional slices and rotates each slice $i$ by an angle $p\theta_i$:

$$\mathbf{R}_{\Theta, p} \mathbf{x} = \begin{pmatrix} \cos(p\theta_1) & -\sin(p\theta_1) & 0 & 0 & \dots \\ \sin(p\theta_1) & \cos(p\theta_1) & 0 & 0 & \dots \\ 0 & 0 & \cos(p\theta_2) & -\sin(p\theta_2) & \dots \\ 0 & 0 & \sin(p\theta_2) & \cos(p\theta_2) & \dots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix} \mathbf{x}$$

Where $\theta_i = \text{base}^{-2(i-1)/d}$.

Although the raw outputs of the Key projection layer ($x W_K$) live in a highly structured, low-rank subspace, the position-dependent rotation $\mathbf{R}_{\Theta, p}$ rotates each token's key vector by a *different* set of angles. When keys from multiple token positions are stacked together into a sequence matrix $\mathbf{K} \in \mathbb{R}^{T \times d}$, this continuous coordinate rotation spreads the variance across all dimensions. 

This artificially inflates the rank of the matrix, **masking the underlying low-rank structure** from linear projection algorithms like PCA.

#### The Solution: Unapplying RoPE
To expose the true low-rank manifold, the coordinate transformation must be undone on each token Key before performing PCA. 

Using the "rotate-half" formulation where the vector is divided into its first half $\mathbf{a}$ and second half $\mathbf{b}$ ($\mathbf{k} = [\mathbf{a}, \mathbf{b}]$), the inverse RoPE rotation is computed exactly as:

$$\mathbf{k}_{\text{no-rope}} = \mathbf{k} \odot \cos(\theta_p) - \text{rotate\_half}(\mathbf{k}) \odot \sin(\theta_p)$$

Where:
$$\text{rotate\_half}([\mathbf{a}, \mathbf{b}]) = [-\mathbf{b}, \mathbf{a}]$$

By applying this inverse rotation, all keys are aligned back to a stationary, non-rotating reference coordinate frame. Once aligned, the singular values collapse back to their low-rank structure, making linear dimension reduction highly effective.

---

### 4. Principal Component Analysis (PCA) & Singular Value Decay on K

Once the RoPE rotation is unapplied, we can perform Principal Component Analysis (PCA) on the aligned Key cache to find the low-dimensional projection axes.

#### The Mathematical Formulation
Let $\mathbf{K}_{\text{no-rope}} \in \mathbb{R}^{N \times D}$ be the matrix of aligned key vectors across a batch of $N$ tokens.
1.  Compute the mean vector $\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{k}_i$.
2.  Center the data: $\mathbf{\bar{K}} = \mathbf{K}_{\text{no-rope}} - \mathbf{\mu}$.
3.  Execute low-rank Singular Value Decomposition (using randomized solvers for speed):
    $$\mathbf{\bar{K}} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where:
*   $\mathbf{U} \in \mathbb{R}^{N \times r}$ contains the left-singular vectors (the compressed representation coefficients, or coordinates).
*   $\mathbf{\Sigma} \in \mathbb{R}^{r \times r}$ is the diagonal matrix of the top $r$ singular values ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r$).
*   $\mathbf{V} \in \mathbb{R}^{D \times r}$ contains the right-singular vectors (the **PCA basis** or projection axes).

#### The Sharp Elbow Phenomenon
The "sharp elbow" refers to the rate of decay of the singular values $\sigma_i$. If you plot $\sigma_i$ against the component index $i$, you observe a steep cliff:

```text
       Singular Value σ_i
           ▲
           │█ 
           │██
           │███
           │████  ◄── Sharp Elbow (Truncation Point r ≈ 192)
           │██████▒▒░░░░
           └──────────────────────────► Component Index i (1..1024)
```

In Layer 12 of Llama-3.1-8B (where the ambient dimension $D = 1024$), the first $192$ components capture over $99.5\%$ of the total Frobenius norm (variance) of the matrix once RoPE is unapplied. 

This mathematical "elbow" guarantees that we can truncate the projection to $r=192$ dimensions, discarding the remaining $832$ dimensions with negligible loss of semantic accuracy.

---

```text
Please say "continue" to proceed to the next set of terms (The 4x Drop Penalty, Hadamard Rotations, Vector Quantization, and Attention Sinks).
```
