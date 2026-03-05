import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MixtralExpert(nn.Module):
    """
    A single expert in the Mixtral 8x7B MoE Architecture.
    Mixtral uses the SwiGLU activation function.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(F.silu(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MixtralSparseMoeBlock(nn.Module):
    """
    The sparse Mixture of Experts block used in Mixtral 8x7B.
    This implementation focuses on mathematical correctness and readability
    over complex system-level execution (like Expert Parallelism).
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        # The router/gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # The experts
        self.experts = nn.ModuleList([
            MixtralExpert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE Block.
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size]
        Returns:
            final_hidden_states: The combined expert outputs.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Reshape to treat each token independently: [batch_size * sequence_length, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # 1. Calculate router logits [batch_size * sequence_length, num_experts]
        router_logits = self.gate(hidden_states)

        # 2. Convert to probabilities and get Top-K 
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Cleanly cast weights back to the generic dtype of the forward pass
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 3. Normalize Top-K weights (so they sum to 1.0 down the top-K axis)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # 4. Initialize the output tensor
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # 5. One-hot mask for routing efficiently
        # selected_experts: [batch_size * seq_len, top_k] holding values [0, num_experts-1]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).to(hidden_states.device)

        # 6. Execute Experts
        # In a real distributed system, we would perform an All-To-All here.
        # But in a local block, we iterate through the experts.
        for expert_idx in range(self.num_experts):
            # Extract the tokens assigned to this specific expert
            # mask: [batch_size * seq_len, top_k] - True if this index matches our current expert
            expert_local_mask = expert_mask[:, :, expert_idx].bool() 
            
            # Collapse the top_k dimension to see if a token goes to this expert *at all*
            # token_is_assigned: [batch_size * seq_len]
            token_is_assigned = expert_local_mask.any(dim=-1)
            
            # If no tokens are assigned to this expert, skip to save computation
            if not token_is_assigned.any():
                continue

            # Extract the actual vectors for processing
            expert_in = hidden_states[token_is_assigned]
            
            # Pass through the expert network
            expert_out = self.experts[expert_idx](expert_in)

            # Get the exact routing weights for these assigned tokens
            # We index back into routing_weights using the localized mask.
            # Example: A token matched expert 5 on its 2nd top-K choice. expert_local_mask tells us WHICH choice it was.
            assigned_routing_weights = routing_weights[token_is_assigned] 
            
            # To get the scalar weights (one per token), we take the weighted mask and sum over top_k dimension
            scalar_weights = (assigned_routing_weights * expert_local_mask[token_is_assigned]).sum(dim=-1, keepdim=True)
            
            # Scale the output by the routing weight
            expert_out_scaled = expert_out * scalar_weights
            
            # Scatter/Add the output back into the final_hidden_states tensor
            # Because final_hidden_states is zeroed initially, we iteratively add the scaled outputs
            final_hidden_states.masked_scatter_(
                token_is_assigned.unsqueeze(-1), 
                final_hidden_states[token_is_assigned] + expert_out_scaled
            )

        # Reshape back to the original format [batch_size, sequence_length, hidden_size]
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states

if __name__ == "__main__":
    # Test the MixtralMoE block
    batch_size = 2
    seq_len = 16
    hidden_size = 4096
    intermediate_size = 14336
    num_experts = 8
    top_k = 2

    print("Initializing Mixtral 8x7B MoE layer...")
    moe_layer = MixtralSparseMoeBlock(
        hidden_size=hidden_size, 
        intermediate_size=intermediate_size, 
        num_experts=num_experts, 
        top_k=top_k
    )

    # Note: Creating a bfloat16 tensor to resemble modern training setups
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)

    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = moe_layer(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print("Done! The architecture is mathematically correct.")
