You already have these foundations:

```text
A. Shape literacy
B. Gradients, loss, backprop
C. Tokenization, embeddings, positions
D. Full Transformer block
```

Next foundations I’d add, in this order:

# Foundation E: Language Modeling Objective

You need this before CS336-style work.

Covers:

```text
next-token prediction
logits over vocabulary
cross-entropy
perplexity
teacher forcing
causal masking
train vs validation loss
```

Core question:

> How does a Transformer become a language model instead of just a stack of tensor operations?

# Foundation F: Optimization and Training Dynamics

You have gradients, but not training behavior.

Covers:

```text
SGD vs AdamW
learning rate
warmup
weight decay
gradient clipping
batch size
gradient accumulation
loss curves
overfitting / underfitting
training instability
```

Core question:

> Once gradients exist, how do we actually train without blowing up or fooling ourselves?

# Foundation G: Tensor Memory and Compute

This is essential for understanding Shard, KV cache, FlashAttention, batching, and inference.

Covers:

```text
parameters vs activations
FLOPs
memory bandwidth
HBM vs SRAM/cache
batching
prefill vs decode
KV cache size
attention cost O(T²)
arithmetic intensity
```

Core question:

> Why is the bottleneck often memory movement, not math?

# Foundation H: Inference and Decoding

This turns language modeling into actual generation.

Covers:

```text
greedy decoding
sampling
temperature
top-k
top-p
repetition
prefill
decode loop
KV cache
latency vs throughput
streaming output
```

Core question:

> How does a trained model actually generate text token by token?

# Foundation I: Evaluation and Benchmarking

You need this if you want to judge papers or systems.

Covers:

```text
held-out evaluation
perplexity
accuracy/F1/ROUGE
NIAH
LongBench-style evals
contamination
ablations
baselines
statistical noise
latency/memory/quality tradeoffs
```

Core question:

> How do I know this method actually works?

# Foundation J: Numerical Precision and Quantization

This connects directly to modern inference work.

Covers:

```text
fp32 / fp16 / bf16
int8 / int4
scale/zero point
symmetric vs asymmetric quantization
per-tensor vs per-channel
NF4
vector quantization
Hadamard rotations
quantization error
```

Core question:

> What happens when we store numbers with fewer bits?

# Foundation K: GPU Kernels and Systems Basics

Not full CUDA mastery. Just enough to read systems papers.

Covers:

```text
kernel launches
fusion
Triton
tiling
matmul kernels
FlashAttention idea
memory coalescing
why Python overhead matters
profiling
```

Core question:

> Why does the same math run fast or slow depending on implementation?

# Foundation L: Data Pipelines

Often skipped, but crucial.

Covers:

```text
datasets
deduplication
token budgets
sequence packing
train/val splits
data quality
mixtures
contamination
streaming data
```

Core question:

> What exactly is the model learning from?

# Foundation M: Model Adaptation

You already touched LoRA, but make it its own foundation.

Covers:

```text
pretraining
fine-tuning
SFT
LoRA
adapters
RLHF / DPO basics
verifier/reward models
catastrophic forgetting
```

Core question:

> How do we change a pretrained model’s behavior without retraining everything?

# My recommended next four

Do these next:

```text
E. Language Modeling Objective
F. Optimization and Training Dynamics
G. Tensor Memory and Compute
H. Inference and Decoding
```

Those will make CS336, Shard, KV cache compression, and model training papers much more readable.

After that:

```text
I. Evaluation
J. Quantization
K. GPU kernels
L. Data pipelines
M. Adaptation
```

The immediate next lesson should be:

> **Foundation E: Language Modeling Objective**

