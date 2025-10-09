# 4️⃣ CNN for Text Classification
# TextCNN Example in PyTorch (No torchtext dependency)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np

# -------------------------------
# 1. Sample Dataset
# -------------------------------
texts = [
    "I love this movie",
    "This film was terrible",
    "What a great movie",
    "I hated this movie",
    "Amazing storyline and acting",
    "The plot was boring",
    "Fantastic direction",
    "Worst movie ever"
]

labels = ["positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# -------------------------------
# 2. Tokenization and Vocabulary
# -------------------------------
def tokenize(text):
    return text.lower().split()

# Build vocabulary manually
counter = Counter()
for t in texts:
    counter.update(tokenize(t))

vocab_list = ["<unk>"] + list(counter.keys())
vocab = {word: idx for idx, word in enumerate(vocab_list)}

def text_to_ids(tokens):
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")

# Convert texts to sequences of indices
max_len = 10  # max sequence length
def encode_text(text):
    tokens = tokenize(text)
    token_ids = text_to_ids(tokens)
    # Pad or truncate
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids

X = np.array([encode_text(t) for t in texts])

# -------------------------------
# 3. Dataset and DataLoader
# -------------------------------
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# -------------------------------
# 4. Model Definition
# -------------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes)*num_filters, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, emb_dim]
        convs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        out = torch.cat(pools, 1)
        return self.fc(out)

# Instantiate model
embed_dim = 50
num_classes = 2
model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)

# -------------------------------
# 5. Training Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# -------------------------------
# 6. Training Loop
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -------------------------------
# 7. Evaluation
# -------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()


#Result 
# Vocabulary Size: 23
# Epoch 1/5, Loss: 0.8045
# Epoch 2/5, Loss: 0.4885
# Epoch 3/5, Loss: 0.3896
# Epoch 4/5, Loss: 0.3103
# Epoch 5/5, Loss: 0.1703