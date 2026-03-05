# Lecture 4: History, Trends, and the Current Landscape of MoE Models

Mixture-of-Experts has exploded in popularity recently due to its success in Large Language Models (LLMs), but the underlying concept is decades old. This lecture explores the historical evolution of MoE, how it arrived at its current SOTA status, and the prevailing trends in architecture design.

## 1. The Early Days (1990s)

The theoretical foundation of MoE was established long before the deep learning boom.

**"Adaptive Mixtures of Local Experts" (Jacobs, Jordan, Nowlan, & Hinton, 1991)**
*   This seminal paper introduced the core idea: instead of training one large global model to solve a complex problem, train a system of multiple "local" expert models, governed by a "gating network" that decides which expert to trust for a given input.
*   The original motivation was **divide-and-conquer**: forcing different neural networks to learn different sub-spaces of the data distribution.
*   These early models were small, typically ensembling Support Vector Machines (SVMs) or very shallow neural networks.

## 2. Bringing MoE to Deep Learning (2017)

For a long time, MoE was considered a niche ensemble method. The breakthrough that brought it into the modern deep learning era was focused on *computational scale*.

**"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al., 2017)**
*   Noam Shazeer and colleagues at Google Brain demonstrated how to integrate MoE into deep RNNs (LSTMs) for language modeling and machine translation.
*   **The Innovation:** They introduced the **Sparsely-Gated** mechanism (noisy Top-K routing). Before this, gating was dense (all experts ran, and outputs were weighted). By enforcing sparsity (e.g., Top-2 out of 1024 experts), they proved that a model's parameter count could scale massively without a commensurate increase in FLOPs.
*   They also introduced the first versions of load-balancing losses to fight the "dead expert" problem.

## 3. The Transformer Era and Switch Transformer (2021)

While the 2017 work was on LSTMs, the architecture world quickly moved to Transformers.

**"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., Google, 2021)**
*   This paper proved that MoE could be highly effective when applied to the Feed-Forward Network (FFN) layers of a Transformer.
*   **The Innovation (Switch Routing):** They simplified routing to **Top-1** (a token is sent to only *one* expert). This drastically reduced communication overhead and computation, allowing them to train a 1.6 Trillion parameter model (which was mind-boggling at the time).
*   They codified the modern load-balancing loss and capacity factor math used today.

## 4. The Open-Weight LLM Explosion (2023-Present)

Until late 2023, MoE was mostly a technique used in closed-source giants (it is widely rumored that GPT-4 is an 8x220B MoE model). The landscape shifted dramatically with the release of open-weight MoE models.

### Mixtral 8x7B (Mistral AI, Dec 2023)
*   Mixtral proved that MoE could be run locally by the community. It used 8 experts, routing each token to the Top-2.
*   It achieved performance comparable to Llama-2 70B while only utilizing ~13B *active* parameters during inference, proving the immense efficiency gains.

### DeepSeek-MoE (DeepSeek AI, Jan 2024)
DeepSeek-MoE introduced a major architectural paradigm shift regarding expert granularity and shared knowledge.
*   **Fine-Grained Experts:** Instead of 8 large experts like Mixtral, DeepSeek uses many smaller experts (e.g., 64). A token might be routed to Top-8. Why? smaller experts can specialize much more aggressively on narrow topics, while selecting 8 out of 64 provides a massive combinatorial space of routing paths.
*   **Shared Experts:** They realized that some knowledge is universal (e.g., basic grammar, structural tokens). If every expert has to learn grammar, it wastes capacity. DeepSeek introduced "Shared Experts"—FFNs that are *always* executed for every token, regardless of the router. The routed experts then only focus on highly specialized semantic knowledge.

### Qwen1.5 / Qwen2 MoE (Alibaba, 2024)
Alibaba adopted the fine-grained MoE approach, proving its robustness. They introduced models like Qwen1.5-32B-MoE.

### DBRX (Databricks, 2024)
Databricks released a 132B parameter MoE (36B active) that uses a very fine-grained 16-expert system (choosing top-4).

## 5. Current Trends and the Future

1.  **Extreme Fine-Grained Routing:** The trend is moving away from a few massive experts (like Mixtral 8x) toward dozens or hundreds of small experts.
2.  **Expert Choice Routing:** Moving away from tokens choosing experts, to experts choosing tokens to ensure perfect load balancing on hardware.
3.  **MoE in Vision and Multimodal:** Architectures like V-MoE (Vision MoE) are applying these exact routing principles to Vision Transformers (ViTs) for image classification and generation tasks.
4.  **Hardware Co-Design:** Nvidia's Blackwell architecture and advanced Triton kernels are being explicitly designed to accelerate the memory-bound, sparse memory access patterns required by MoE networks.

---
**Conclusion:** MoE has evolved from a niche ensemble technique into the defining architectural paradigm of the post-GPT-4 era, offering the only known path to continuous scaling without hitting an insurmountable compute wall.
