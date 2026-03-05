import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A simple Feed-Forward Network expert."""
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class TopKRouter(nn.Module):
    """
    Standard Top-K Router for MoE.
    Returns routing weights, selected expert indices, and the auxiliary load balancing loss.
    """
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # (B*S, d_model)
        
        # 1. Compute routing probabilities
        logits = self.gate(x_flat) # (B*S, num_experts)
        probs = F.softmax(logits, dim=-1) # (B*S, num_experts)
        
        # 2. Select Top-K experts
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1) # (B*S, top_k)
        
        # Normalize the Top-K probabilities to sum to 1 over the selected experts
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True) # (B*S, top_k)
        
        # 3. Calculate Auxiliary Load Balancing Loss
        # a. Fraction of tokens routed to each expert
        # Create a boolean mask of which experts were selected
        selected_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1) # (B*S, num_experts)
        # Fraction of total tokens
        f_i = selected_mask.float().mean(dim=0) # (num_experts,)
        
        # b. Average routing probability given to each expert
        p_i = probs.mean(dim=0) # (num_experts,)
        
        # c. Multiplicative load balancing loss
        aux_loss = self.num_experts * torch.sum(f_i * p_i)
        
        return topk_weights, topk_indices, aux_loss


class SparseMoELayer(nn.Module):
    """
    A Sparse Mixture of Experts Layer.
    """
    def __init__(self, d_model, d_hidden, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Routing
        routing_weights, selected_experts, aux_loss = self.router(x)
        
        # We need to compute the output.
        # For a standard implementation (like Mixtral), we dispatch tokens to experts.
        # For simplicity in this toy model, we will loop over experts and process tokens assigned to them.
        
        final_output = torch.zeros_like(x_flat)
        
        # Iterate over each expert
        for expert_idx, expert_layer in enumerate(self.experts):
            # Find tokens assigned to this expert
            # selected_experts is (B*S, top_k), we need to find where expert_idx is present
            token_mask = (selected_experts == expert_idx)
            
            # token_indices: shape (num_tokens_for_expert,)
            token_indices = torch.nonzero(token_mask.sum(dim=-1)).squeeze(-1)
            
            if token_indices.numel() > 0:
                # Extract the tokens for this expert
                expert_inputs = x_flat[token_indices]
                
                # Pass through the expert
                expert_outputs = expert_layer(expert_inputs)
                
                # Get the routing weights for these specific tokens and this expert
                # token_mask[token_indices] will show WHICH top_k slot this expert occupied
                slot_indices = torch.nonzero(token_mask[token_indices])[:, 1]
                weights = routing_weights[token_indices, slot_indices].unsqueeze(-1)
                
                # Scale and accumulate
                final_output[token_indices] += expert_outputs * weights
                
        return final_output.view(batch_size, seq_len, d_model), aux_loss


class ToyMoEModel(nn.Module):
    """
    A toy model replacing standard FFN with MoE.
    """
    def __init__(self, vocab_size, d_model, d_hidden, num_experts, top_k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Toy Self Attention layer (not physically realistic, just for flow)
        self.attention = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.moe_layer = SparseMoELayer(d_model, d_hidden, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        h = self.embedding(x)
        
        # Attention block
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        
        # MoE block
        moe_out, aux_loss = self.moe_layer(h)
        h = self.norm2(h + moe_out)
        
        # Output prediction
        logits = self.lm_head(h)
        return logits, aux_loss

if __name__ == "__main__":
    # Quick sanity check
    model = ToyMoEModel(vocab_size=1000, d_model=128, d_hidden=256, num_experts=4, top_k=2)
    inputs = torch.randint(0, 1000, (2, 10)) # Batch size 2, Seq len 10
    logits, aux_loss = model(inputs)
    print(f"Logits shape: {logits.shape}")
    print(f"Auxiliary Loss: {aux_loss.item()}")
