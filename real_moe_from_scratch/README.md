# Implementing a Real MoE Architecture: Mixtral 8x7B

This directory contains `mixtral_moe.py`, a clean, from-scratch PyTorch implementation of the Sparse Mixture of Experts block used in the Mixtral 8x7B model.

## Core Components

1.  **`MixtralExpert`**: Rather than a standard MLP, Mixtral employs a `SwiGLU` activation-based feed-forward network, which uses 3 separate weight matrices (`w1`, `w2`, `w3`). This is standard across LLaMA and other modern architectures.
2.  **`MixtralSparseMoeBlock`**: The MoE routing layer itself.

## The Routing Logic Breakdown

Understanding how tokens are parsed through python tensors is notoriously tricky. Our implementation utilizes a clean masking approach:

1.  **Linear Projection:** `self.gate(hidden_states)` calculates a score for every expert.
2.  **Softmax & TopK:** We compute the Top-2 highest scores per token and normalize.
3.  **One-Hot Mask:** A crucial tensor `expert_mask` is generated. It tells us exactly which top-$k$ choice corresponds to which expert ID.
4.  **Iteration:** We iterate over the 8 experts. For each expert, we isolate only the tokens routed to it, pass them through the `SwiGLU` network, scale the outputs by the router probability, and scatter the results back into a zero-initialized full tensor.

## Running the Code

You can test the implementation by running:

```bash
python mixtral_moe.py
```

This will initialize an MoE layer matching Mixtral's dimensions (hidden size `4096`, intermediate size `14336`) and pass a dummy tensor through it to verify tensor dimensionality.

## Limitations of this Implementation

This script focuses on mathematical and architectural correctness. It will run successfully on a single GPU or CPU.

However, it is *not* optimized for distributed high-performance training. Real implementations (found in vLLM or Megatron-LM) handle:
*   **Expert Parallelism**: Instead of a python `for` loop over experts, different experts reside on different GPUs.
*   **Triton/CUDA Kernels**: Advanced custom kernels are used to dynamically pack tokens and compute the sparse matrix multiplications without physically moving contiguous memory chunks (e.g. `grouped_gemm`).
*   **Auxiliary Load Balancing**: Models like Mixtral add specific auxiliary losses during training, which would be injected into the backward graph here.
