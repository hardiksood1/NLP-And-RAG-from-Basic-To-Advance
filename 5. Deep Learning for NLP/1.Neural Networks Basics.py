import torch
import torch.nn as nn
import torch.optim as optim

# Fake dataset: (sentence length 3, vocab size 10)
X = torch.randint(0, 10, (5, 3))   # 5 sentences, each 3 words
y = torch.tensor([0, 1, 0, 1, 0]) # labels (0=neg, 1=pos)

class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  
        self.fc1 = nn.Linear(embed_dim*3, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.embed(x)          # shape: [batch, seq_len, embed_dim]
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN(vocab_size=10, embed_dim=8, hidden_dim=16, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (1 epoch)
for epoch in range(5):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# output
# Epoch 1, Loss: 0.7079
# Epoch 2, Loss: 0.5560
# Epoch 3, Loss: 0.4319
# Epoch 4, Loss: 0.3354
# Epoch 5, Loss: 0.2628 
