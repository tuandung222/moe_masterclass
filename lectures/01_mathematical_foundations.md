# Lecture 1: Mathematical Foundations of Mixture-of-Experts (MoE)

Welcome to the **MoE Masterclass**. This series is designed for AI researchers and practitioners with backgrounds in mathematics, computer vision (CV), and large language models (LLMs). Our goal is to gain an in-depth understanding of the Mixture-of-Experts architecture.

## 1. Introduction

The scale of neural networks has been a primary driver of recent AI breakthroughs. However, scaling up entails a significant increase in computational cost. **Mixture-of-Experts (MoE)** is a paradigm that decouples the parameter count of a model from its computational requirement. Instead of activating the entire network for every input token, an MoE model routes each token to a subset of specialized subnetworks called **experts**.

### Concept Overview
Traditional deep learning layers are *dense*—every parameter is used to process every input. An MoE layer replaces a dense layer (typically a Feed-Forward Network or FFN in a Transformer) with:
1.  A set of $N$ independent expert networks $\{E_1, E_2, \dots, E_N\}$.
2.  A trainable **gating network** (or router) $G$ that determines the contribution of each expert to the final output for a given input $x$.

## 2. Mathematical Formulation

Let $x \in \mathbb{R}^d$ be the input representation of a token. The output of the MoE layer $y$ is computed as a weighted sum of the expert outputs:

$$ y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x) $$

Where:
*   $N$: The total number of experts.
*   $E_i(x): \mathbb{R}^d \rightarrow \mathbb{R}^h$: The output of the $i$-th expert. Usually, all experts share the same architectural structure but have independent weights.
*   $G(x)_i$: The $i$-th scalar component of the router's output vector, representing the probability or weight assigned to the $i$-th expert.

### The Gating Function

The gating network $G(x)$ is typically a parameterized function, most commonly a linear projection followed by a Softmax normalization to ensure the routing weights sum to 1. Let $W_g \in \mathbb{R}^{d \times N}$ be the weight matrix of the router:

$$ G(x) = \text{Softmax}(x \cdot W_g) $$

$$ G(x)_i = \frac{\exp(x \cdot W_{g, i})}{\sum_{j=1}^N \exp(x \cdot W_{g, j})} $$

Where $W_{g, i}$ is the $i$-th column of the router weight matrix.

### Sparsity via Top-K Gating

The formulation above describes a *soft* or *dense* mixture, where every expert processes the input, and their outputs are combined. While this increases model capacity, it does *not* save computation.

To achieve sparse computation, we enforce that $G(x)_i = 0$ for all but the top $k$ experts, where $k \ll N$ (e.g., $k=1$ or $k=2$ for $N=8$). We define a sparse gating mechanism:

$$ G_{\text{TopK}}(x)_i = \begin{cases} \frac{\exp(H(x)_i)}{\sum_{j \in \mathcal{T}} \exp(H(x)_j)} & \text{if } i \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases} $$

Where $H(x) = x \cdot W_g$, and $\mathcal{T}$ is the set of indices of the top-$k$ elements of $H(x)$.

## 3. Probabilistic Interpretation

We can view MoE through the lens of a **Latent Variable Model**. Let $E$ be a discrete latent variable representing the choice of expert, taking values in $\{1, 2, \dots, N\}$.

1.  **Prior / Routing Strategy**: The router output $G(x)_i$ models the probability of selecting expert $i$ given the input $x$:
    $$ P(E=i | x) = G(x)_i $$
2.  **Likelihood / Expert Specialization**: The expert output $E_i(x)$ can be seen as the expected output given that expert $i$ was chosen.

The final output is the expectation of the expert outputs under the routing distribution:
$$ \mathbb{E}_{E \sim G(x)}[E_E(x)] = \sum_{i=1}^N P(E=i | x) \cdot E_i(x) = y $$

### Why this matters

This probabilistic perspective highlights a crucial dynamic during training: **Expert Specialization vs. Representation Collapse**.
*   Ideally, different regions of the input space are assigned to different experts, allowing each subnet to specialize.
*   However, if the router strongly prefers a single expert early in training (due to random initialization or a lucky update), that expert gets more gradient updates, becomes better at its task, and the router prefers it even more. This creates a feedback loop leading to **Representation Collapse**, where only a few experts are ever utilized. This necessitates auxiliary routing losses, which we will cover in Lecture 2.

## 4. Optimization and Gradients

When using sparse gating (Top-K), the routing operation introduces a non-differentiable step: the `topk` selection.

Through the Top-K masking, the gradient flows only to the selected experts.
$$ \frac{\partial y}{\partial E_i} = \begin{cases} G(x)_i & \text{if } i \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases} $$

The gradient with respect to the routing logits $H(x)$ (and consequently router weights $W_g$) flows only through the non-zero routing weights:
$$ \frac{\partial y}{\partial H(x)_i} = \frac{\partial G_{\text{TopK}}(x)_i}{\partial H(x)_i} \cdot E_i(x) \quad \text{for } i \in \mathcal{T} $$

### The "Dead Expert" Problem

Since gradients only flow to experts that receive tokens, an expert that is never selected by the Top-K operation will receive zero gradients. Its weights will never update, and it will remain "dead." Addressing this requires careful initialization, load balancing techniques, and novel routing paradigms (like Expert Choice routing).

---
**Next:** In Lecture 2, we will dive deep into Load Balancing, auxiliary losses, and modern routing mechanisms.
