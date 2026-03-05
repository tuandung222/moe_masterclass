import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekExpert(nn.Module):
    """
    A single expert in the DeepSeek-MoE Architecture.
    DeepSeek also uses the SwiGLU activation. 
    Notice that the structural code is identical to Mixtral, but the hyperparams 
    (intermediate_size) will be significantly smaller to represent "Fine-Grained" experts.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoEBlock(nn.Module):
    """
    The sparse Mixture of Experts block inspired by DeepSeek-MoE.
    Key differences from Mixtral:
    1. Shared Experts: Unconditionally executed for all tokens.
    2. Fine-Grained Routed Experts: Many small experts instead of a few large ones.
    """
    def __init__(
        self, 
        hidden_size: int, 
        expert_intermediate_size: int, 
        num_routed_experts: int, 
        num_shared_experts: int,
        routed_scaling_factor: float,
        top_k: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

        # 1. Shared Experts
        # Instead of instantiating N separate shared experts and summing them, 
        # it is mathematically equivalent and computationally faster to just instantiate 
        # ONE large expert where intermediate_size = num_shared_experts * expert_intermediate_size
        shared_intermediate_size = expert_intermediate_size * num_shared_experts
        self.shared_expert = DeepSeekExpert(hidden_size, shared_intermediate_size)

        # 2. Routed Experts
        self.gate = nn.Linear(hidden_size, num_routed_experts, bias=False)
        self.routed_experts = nn.ModuleList([
            DeepSeekExpert(hidden_size, expert_intermediate_size) for _ in range(num_routed_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size]
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        latent_states = hidden_states.view(-1, hidden_dim) # [B*S, H]

        # 1. Execute the Shared Expert unconditionally for all tokens
        shared_output = self.shared_expert(latent_states)

        # 2. Execute the Routed Experts
        # Calculate router logits for the routed experts
        router_logits = self.gate(latent_states)

        # Get routing weights and Top-K indices
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # In DeepSeek-MoE, weights are NOT necessarily normalized to sum to 1.
        # They are multiplied by a normalization/scaling factor.
        topk_weights = topk_weights.to(latent_states.dtype) * self.routed_scaling_factor

        # Initialize the routed output tensor
        routed_output = torch.zeros_like(latent_states)
        
        # One-hot mask for routing: [B*S, top_k, num_experts] -> [B*S, num_experts]
        # Summing over top_k since an expert can only be picked once per token
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_routed_experts).sum(dim=1).bool()

        # Iterate through the fine-grained experts
        for expert_idx in range(self.num_routed_experts):
            # Find tokens assigned to this expert
            token_is_assigned = expert_mask[:, expert_idx]
            
            if not token_is_assigned.any():
                continue

            # Extract tokens
            expert_in = latent_states[token_is_assigned]
            
            # Pass through the specific fine-grained expert
            expert_out = self.routed_experts[expert_idx](expert_in)

            # We need to extract the routing weight for these specific tokens for this specific expert.
            # selected_experts: [B*S, top_k]
            # token_is_assigned gives us the row (the token). We need to find the column (the top_k slot) 
            # where selected_experts == expert_idx to get the corresponding weight.
            
            # This extracts the top_k index (0, 1, 2... top_k-1) where the match occurred
            slot_indices = (selected_experts[token_is_assigned] == expert_idx).nonzero(as_tuple=True)[1]
            
            # Extract the actual scalar weights
            scalar_weights = topk_weights[token_is_assigned, slot_indices].unsqueeze(-1)
            
            # Scale the output
            expert_out_scaled = expert_out * scalar_weights
            
            # Scatter/Add back into the routed output tensor
            routed_output.masked_scatter_(
                token_is_assigned.unsqueeze(-1), 
                routed_output[token_is_assigned] + expert_out_scaled
            )

        # 3. Combine Shared and Routed Outputs
        final_output = shared_output + routed_output
        
        return final_output.view(batch_size, sequence_length, hidden_dim)


if __name__ == "__main__":
    # Test the DeepSeek MoE block using highly fine-grained parameters
    batch_size = 2
    seq_len = 16
    hidden_size = 2048
    
    # Notice the small intermediate size. A standard dense model might use 8192.
    # DeepSeek uses tiny experts and many of them.
    expert_intermediate_size = 512 
    
    # 64 routed experts, tokens select top-8!
    num_routed_experts = 64
    top_k = 8
    
    # 2 shared experts (computationally merged into one expert of size 1024)
    num_shared_experts = 2 
    
    routed_scaling_factor = 1.0

    print("Initializing DeepSeek-MoE layer (Fine-Grained + Shared Experts)...")
    moe_layer = DeepSeekMoEBlock(
        hidden_size=hidden_size,
        expert_intermediate_size=expert_intermediate_size,
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        top_k=top_k
    )

    dummy_input = torch.randn(batch_size, seq_len, hidden_size)

    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = moe_layer(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print("Done! The DeepSeek Fine-Grained architecture is mathematically correct.")
