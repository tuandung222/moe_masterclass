import torch
import torch.nn as nn
import torch.optim as optim
from toy_moe import ToyMoEModel

def generate_synthetic_data(num_samples, seq_len, vocab_size):
    """Generates random sequence data for next-token prediction."""
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Y is simply shifted X
    Y = torch.cat([X[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=-1)
    return X, Y

def train():
    # Hyperparameters
    vocab_size = 500
    d_model = 64
    d_hidden = 128
    num_experts = 4
    top_k = 2
    batch_size = 16
    seq_len = 20
    num_epochs = 100
    learning_rate = 1e-3
    aux_loss_coef = 0.01

    model = ToyMoEModel(vocab_size, d_model, d_hidden, num_experts, top_k)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Synthetic Dataset
    X_train, Y_train = generate_synthetic_data(1000, seq_len, vocab_size)

    print("Starting Training of Toy MoE Model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        
        # Simple rudimentary batching
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = Y_train[i:i+batch_size]

            optimizer.zero_grad()

            # Forward pass
            logits, aux_loss = model(batch_x)
            
            # Loss computation
            # Flatten logits and targets for CrossEntropy
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = batch_y.view(-1)
            
            main_loss = criterion(logits_flat, targets_flat)
            
            # Combine losses
            loss = main_loss + aux_loss_coef * aux_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += main_loss.item()
            total_aux_loss += aux_loss.item()
            
        avg_main_loss = total_loss / (len(X_train) / batch_size)
        avg_aux_loss = total_aux_loss / (len(X_train) / batch_size)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Main Loss: {avg_main_loss:.4f} | Aux Loss: {avg_aux_loss:.4f} | Total Loss: {(avg_main_loss + aux_loss_coef*avg_aux_loss):.4f}")

    print("Training Complete!")

if __name__ == "__main__":
    train()
