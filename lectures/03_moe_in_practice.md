# Lecture 3: MoE in Practice - Scaling and Systems

In the previous lectures, we covered the mathematics of routing and the strategies employed to train robust MoE models. In this final lecture of the theoretical series, we analyze how MoE is leveraged in state-of-the-art Large Language Models (LLMs) and the systems engineering required to scale them.

## 1. Architectural Patterns in Modern LLMs

MoE has become the default architectural path for open-weight foundation models exceeding 70B active parameters. Let's look at the anatomical patterns.

### 1.1 Sparse MoE Replacements in Transformers
In a standard Transformer block, the computation is dominated by the Multi-Head Attention (MHA) and the Feed-Forward Network (FFN).

SOTA models (e.g., Mixtral 8x7B, Databricks DBRX, Qwen1.5-MoE) replace the dense FFN layers with Sparse MoE layers.
*   The Attention layers are dense (shared across all tokens).
*   The FFN layers are sparse (tokens routed to experts).

This is because FFNs represent the vast majority of parameters in a Transformer and are highly localized (acting independently on each token's representation), making them the perfect candidate for conditional execution.

### 1.2 Granular vs. Coarse Experts

There's a design trade-off in determining expert size. Should a model have a few massive experts, or many tiny experts?

*   **Coarse-grained (e.g., Mixtral 8x7B):** Defines 8 large experts. A token selects top-2.
*   **Fine-grained (e.g., Qwen MoE, DeepSeek MoE):** Uses fundamentally smaller FFNs but vastly more of them (e.g., 64 experts). A token might select top-4 or top-8.

**The Theory of Fine-Grained MoE:** Research (such as DeepSeekMoE, 2024) suggests that highly granular experts can specialize much more effectively. A smaller FFN is forced to learn a very specific concept, and routing across 64 experts provides a vast combinatorics of active paths ($64 \choose 4$), increasing the effective capacity.

### 1.3 Shared Experts

Recent models introduce the concept of **Shared Experts**. Instead of routing a token entirely to specialized experts, the architecture designates one or more experts as "shared", which means they are *always* unconditionally executed for every token.

**Formula:**
$$ y = E_{shared}(x) + \sum_{i \in \mathcal{T}} G_{\text{TopK}}(x)_i \cdot E_i(x) $$

This pattern ensures that general knowledge (e.g., core syntactic structures, highly frequent facts) is not duplicated across many specialized experts, allowing specialized experts to focus exclusively on niche domain knowledge.

## 2. Distributed Training Regimes

Scaling an MoE model requires complex parallelism strategies due to its memory footprint and routing communication overhead.

### 2.1 Expert Parallelism (EP)
Expert Parallelism is a novel paradigm unique to MoE models.

1.  A standard Transformer relies on **Data Parallelism (DP)** (replicating the model across GPUs to process different batches) and **Tensor Parallelism (TP)** (sharding large matrix multiplications across GPUs).
2.  In **Expert Parallelism**, experts are placed on different GPUs.
    *   GPU 0 holds Attention layers + Expert 0, 1
    *   GPU 1 holds Attention layers + Expert 2, 3
3.  **All-to-All Communication:** During an MoE forward pass, tokens computed on GPU 0 might need to be routed to Expert 3 (which lives on GPU 1). This triggers a massive collective communication operation called an `AllToAll` dispatch. Tokens are gathered, shuffled over the network fabric to their target GPUs, processed by the resident experts, and an `AllToAll` recombination sends them back to their origin.

### 2.2 The Memory Bottleneck in Inference

While MoE is famously efficient for training and generation speed (FLOPs-per-token is low), it is severely bottlenecked by **Memory Bandwidth** during inference.

For memory-bound generation (batch size = 1), the latency of the model is dictated by the time it takes to move weights from HBM (High Bandwidth Memory) to the GPU SM registers.
1.  In a dense model, you must load all parameters once to generate a token.
2.  In an MoE model, even though execution is sparse (e.g., only 2 out of 8 experts are active), **you must still load the parameters of the active experts**. If the active experts change from token to token, or if batch sizes increase, you eventually are forced to load almost the entire model into memory just to serve a batch, negating the sparsity benefits.

This is why MoE models are exceptionally challenging to serve efficiently. Frameworks like vLLM and TensorRT-LLM implement sophisticated CUDA kernels (like grouped GEMMs and custom continuous batching) to mitigate this.

---
**Conclusion:** MoE architectures are the key to breaking the dense scaling laws, offering massive parameter counts with sub-linear compute costs. However, they trade compute efficiency for memory capacity and networking bandwidth challenges.

In the practical section, we will implement these concepts in PyTorch.
