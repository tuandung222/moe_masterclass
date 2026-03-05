# Lecture 2: Routing Mechanisms and Load Balancing

In Lecture 1, we established the core mathematical formulation of the Mixture-of-Experts (MoE) layer. We also identified a fundamental pathological behavior in standard Softmax-based Top-K routing: the tendency for representation collapse and "dead experts".

In this lecture, we will explore advanced routing mechanisms designed to mitigate these issues and understand the mathematics of load balancing.

## 1. Challenges with Standard Top-K Routing

Let a batch of $B$ tokens be denoted as $X \in \mathbb{R}^{B \times d}$. The router assigns each token to $k$ experts out of $N$. The two main issues are:

1.  **Imbalanced Assignment:** The router might assign the majority of tokens to a small subset of experts. This causes:
    *   **Bottlenecks:** In distributed systems where experts reside on different GPUs (Expert Parallelism), overloaded experts create a computational bottleneck, leaving other GPUs idle.
    *   **Dead Experts:** Experts that receive no tokens never update their weights.
2.  **Dropped Tokens:** Due to hardware constraints, implementations typically define an **expert capacity** $C$, calculated as $C = \lfloor \frac{B \cdot k}{N} \cdot f \rfloor$, where $f \geq 1.0$ is the capacity factor. If an expert receives more than $C$ tokens, the excess tokens are **dropped** (their representation usually defaults to a residual connection, effectively bypassing the expert layer).

## 2. Load Balancing through Auxiliary Losses

To force the router to distribute tokens evenly, we add an **Auxiliary Load Balancing Loss** ($L_{aux}$) to the main objective function. Let's analyze the formulation used in models like Switch Transformer and Mixtral.

### 2.1 The Two Components of Balance

A perfectly balanced router satisfies two conditions across a batch:
1.  **Routing Fraction:** The sum of routing probabilities for each expert should be roughly uniform.
2.  **Expert Allocation:** The discrete number of tokens actual sent to each expert should be roughly uniform.

Let's formally define these variables over a batch of $B$ tokens:

*   **Average Routing Probability ($P_i$):** The average probability assigned to expert $i$ across all tokens in the batch.
    $$ P_i = \frac{1}{B} \sum_{j=1}^{B} G(x_j)_i $$
*   **Token Fraction ($f_i$):** The actual fraction of tokens routed to expert $i$. Let $c_i$ be the number of tokens routed to expert $i$.
    $$ f_i = \frac{c_i}{B \cdot k} = \frac{1}{B \cdot k} \sum_{j=1}^{B} \mathbb{I}(i \in \text{TopK}(x_j)) $$
    *(Note: $\mathbb{I}$ is the indicator function)*

### 2.2 The Switch Transformer Loss Formulation

If we solely minimized the variance of $f_i$, the loss would be non-differentiable since $f_i$ relies on the discrete `argmax` (or Top-K) operation. If we solely minimized the variance of $P_i$, the router could assign a high uniform probability to an expert without actually sending tokens to it (due to Top-K clipping).

Therefore, the standard auxiliary loss optimizes the dot product of vectors $f \in \mathbb{R}^N$ and $P \in \mathbb{R}^N$:

$$ L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i $$

*   $\alpha$ is a hyperparameter scaling the loss.
*   The multiplier $N$ ensures that under a perfectly uniform distribution ($f_i = \frac{1}{N}$ and $P_i = \frac{1}{N}$), $L_{aux} = \alpha$.

**Why it works:** To minimize $\sum f_i P_i$, the model must make both distributions close to uniform. Because $\sum f_i = 1$ and $\sum P_i = 1$, the minimum of the dot product occurs when both are uniform vectors $[\frac{1}{N}, \dots, \frac{1}{N}]$. Since $P_i$ is fully differentiable with respect to the router weights, gradients flow smoothly to adjust the routing probabilities.

## 3. Alternative Routing Paradigms

While token-choice Top-K routing with auxiliary loss is standard (used in Mixtral, Qwen MoE), researchers are actively exploring alternative paradigms to simplify or improve routing.

### 3.1 Expert Choice Routing

Introduced by Google (Zhou et al., 2022), **Expert Choice (EC) Routing** flips the assignment logic. Instead of each token choosing its top $k$ experts, **each expert chooses its top $C$ tokens**.

**Mechanism:**
1.  Compute routing probabilities $G \in \mathbb{R}^{B \times N}$ for the entire batch.
2.  For each expert $i$ (each column in $G$), select the top $C$ elements.
3.  Each expert processes its selected $C$ tokens.

**Advantages:**
*   **Perfect Load Balancing:** Every expert processes exactly $C$ tokens. There is no need for an auxiliary load balancing loss.
*   **Variable Token Computation:** A single token might be chosen by zero, one, or multiple experts. Tokens that are easy to process might be dropped (no expert selects them), while complex tokens might be selected by many experts, allowing the model to dynamically allocate computational budget per token.

**Disadvantages:**
*   Complex to implement efficiently in an auto-regressive decoding setting, as we process token-by-token rather than in large batches. It is primarily used for encoder-only models or prefix processing.

### 3.2 Noisy Top-K Gating

Used in earlier models (like Shazeer et al., 2017), this adds Gaussian noise to the logits before computing the Top-K assignment.
$$ H(x)_i = x \cdot W_{g, i} + \text{StandardNormal}() \cdot \text{Softplus}(x \cdot W_{noise, i}) $$
This helps with exploration during early training, preventing the router from prematurely committing to a subset of experts.

### 3.3 Zero-routing / Token dropping

In very large scale training, it's observed that dropping tokens (when an expert capacity is exceeded, or intentionally skipping FFN computation via routing thresholds) doesn't severely harm performance if the dropping rate is small (e.g., < 1%). The residual connection carries the un-processed representation forward.

---
**Next:** In Lecture 3, we will explore the systems and scaling aspects of applying MoE architecture in leading LLMs.
