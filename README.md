# Mixture-of-Experts (MoE) Masterclass

Welcome to the **MoE Masterclass**! This repository provides a comprehensive, technically rigorous set of resources for learning about Mixture-of-Experts architectures, from foundational mathematics to implementation details of state-of-the-art models.

This resource is designed for AI researchers, practitioners, and students with a background in mathematics, deep learning, and Large Language Models (LLMs).

## 🗂️ Repository Structure

The repository is divided into theoretical lectures and practical code implementations:

```text
moe_masterclass/
├── README.md                   # This overview file
├── lectures/                   # Theoretical foundations and concepts
│   ├── 01_mathematical_foundations.md
│   ├── 02_routing_and_load_balancing.md
│   ├── 03_moe_in_practice.md
│   └── 04_history_and_trends.md
├── toy_moe/                    # Simple educational PyTorch implementation
│   ├── toy_moe.py
│   └── train_toy.py
├── real_moe_from_scratch/      # Standard MoE from scratch (Mixtral 8x7B style)
│   ├── mixtral_moe.py
│   └── README.md
└── advanced_moe_from_scratch/  # Advanced MoE from scratch (DeepSeek style)
    ├── deepseek_moe.py
    └── README.md
```

## 📖 Recommended Study Guide (Reading Order)

To get the most out of this masterclass, we recommend following this order:

### Phase 1: Theory and Foundations
Start by reading through the documentation in the `lectures/` folder to understand the "Why" and "How" of MoE.

1.  **[Lecture 1: Mathematical Foundations](lectures/01_mathematical_foundations.md)**: Learn the core formulation, probabilistic interpretation, and the optimization challenges of MoE.
2.  **[Lecture 2: Routing and Load Balancing](lectures/02_routing_and_load_balancing.md)**: Dive into Top-K routing, Expert Choice routing, and the critical auxiliary losses needed to prevent representation collapse.
3.  **[Lecture 3: MoE in Practice](lectures/03_moe_in_practice.md)**: Understand system-level constraints like Expert Parallelism (EP), memory bottlenecks, and how MoE operates in modern LLMs.
4.  **[Lecture 4: History and Trends](lectures/04_history_and_trends.md)**: Trace the evolution of MoE from the 1990s to Switch Transformer, and explore current trends like fine-grained and shared experts.

### Phase 2: Practical Implementations
Once you grasp the theory, explore the code to see how it's built in PyTorch.

5.  **Explore `toy_moe/`**: Review `toy_moe.py` to see the simplest possible implementation of a Top-K router and load-balancing loss. Run `python toy_moe/train_toy.py` to see it learn on synthetic data.
6.  **Study `real_moe_from_scratch/`**: Read the local README, then review `mixtral_moe.py`. This implements the precise "Coarse-Grained" routing and `SwiGLU` expert architecture used in Mixtral 8x7B. Run `python real_moe_from_scratch/mixtral_moe.py` to execute a forward pass.
7.  **Explore `advanced_moe_from_scratch/`**: Read the local README, then dive into `deepseek_moe.py`. This demonstrates the cutting-edge "Fine-Grained" routing and "Shared Experts" paradigm popularized by DeepSeek. Run `python advanced_moe_from_scratch/deepseek_moe.py` to see it in action.

## 🚀 Requirements

To run the PyTorch code in this repository, you only need Python and PyTorch installed. The code is designed to be readable and self-contained, avoiding complex dependencies like `vllm` or `megatron-lm` (which are used in distributed production but obfuscate the core logic).

```bash
pip install torch
```
