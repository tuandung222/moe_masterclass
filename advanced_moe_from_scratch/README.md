# Advanced MoE Architecture: DeepSeek-MoE

This directory contains `deepseek_moe.py`, an implementation of the MoE paradigm introduced by the DeepSeek-V2 and DeepSeek-MoE models. 

This architecture differs drastically from the Mixtral architecture found in the `real_moe_from_scratch` folder.

## Key Innovations

### 1. Fine-Grained Experts
Standard MoEs like Mixtral use "Coarse-Grained" experts. For example, Mixtral has 8 enormous experts, and routes to 2.
DeepSeek posits that this limits specialization. Instead, DeepSeek uses **Fine-Grained** experts. They take the parameters of a large expert and shatter them into many tiny ones.
*   In our test block, we define `num_routed_experts = 64`.
*   A token is routed to `top_k = 8`.
*   Because the experts are small, the computational cost (active parameters) remains exactly the same as routing to 2 large experts, but the combinatorial routing paths explode, allowing much greater representational capacity and targeted specialization.

### 2. Shared Experts
When routing tokens purely to specialized experts, redundant knowledge must be learned. For example, every expert needs to understand basic punctuation and grammar, wasting their specialized parameter budget.

DeepSeek introduces **Shared Experts**.
*   These are a core set of FFN parameters that are **unconditionally executed** for every single token, bypassing the router entirely.
*   This acts as a repository for general knowledge, allowing the routed experts to focus entirely on niche domain knowledge (e.g., Python syntax, historical facts).
*   In our code, we mathematically optimize this by creating one large `shared_expert` that represents the concatenated size of $N$ shared experts.

## The Output Equation

The output of the DeepSeek MoE layer can be written as:

$$ y = E_{shared}(x) + \sum_{i \in \mathcal{T}} \text{ScaleFactor} \cdot G(x)_i \cdot E_i(x) $$

Where $E_{shared}$ runs on every token, and $E_i$ are the tiny specialized experts selected by the Top-K gating function.

## Running the Code

```bash
python deepseek_moe.py
```
