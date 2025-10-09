import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────── Device ───────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────── Toy Dataset ───────────
# Example: "hello" -> "hola" (character-level)
src_sentences = [["h","e","l","l","o"], ["b","y","e"], ["t","e","s","t"]]
trg_sentences = [["H","O","L","A"], ["B","Y","E"], ["T","E","S","T"]]

# Build vocab
src_vocab = {ch:i for i,ch in enumerate(sorted(set(sum(src_sentences,[]))))}
trg_vocab = {ch:i for i,ch in enumerate(sorted(set(sum(trg_sentences,[]))))}
src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)

# Convert sentences to indices
def encode(sentences, vocab):
    return [torch.tensor([vocab[ch] for ch in s], dtype=torch.long) for s in sentences]

src_encoded = encode(src_sentences, src_vocab)
trg_encoded = encode(trg_sentences, trg_vocab)

# ─────────── Seq2Seq Models ───────────
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        
    def forward(self, src):
        embedded = self.embedding(src.unsqueeze(0))  # [batch=1, seq_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0).unsqueeze(0)      # [batch=1, seq_len=1]
        embedded = self.embedding(input)             # [1,1,emb_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))      # [batch=1, output_dim]
        return prediction, hidden, cell

# Hyperparameters
EMB_DIM = 16
HIDDEN_DIM = 32
N_EPOCHS = 200
LEARNING_RATE = 0.01

encoder = Encoder(src_vocab_size, EMB_DIM, HIDDEN_DIM).to(device)
decoder = Decoder(trg_vocab_size, EMB_DIM, HIDDEN_DIM).to(device)

# Optimizer and Loss
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ─────────── Training Loop ───────────
for epoch in range(N_EPOCHS):
    epoch_loss = 0
    for src_seq, trg_seq in zip(src_encoded, trg_encoded):
        src_seq, trg_seq = src_seq.to(device), trg_seq.to(device)
        optimizer.zero_grad()
        
        hidden, cell = encoder(src_seq)
        
        loss = 0
        input_decoder = trg_seq[0]  # start with first character
        for t in range(1, len(trg_seq)):
            output, hidden, cell = decoder(input_decoder, hidden, cell)
            loss += criterion(output, trg_seq[t].unsqueeze(0))
            input_decoder = trg_seq[t]  # teacher forcing
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(trg_seq)
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(src_encoded):.4f}")

# ─────────── Inference ───────────
def translate(sentence, max_len=10):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        src_seq = torch.tensor([src_vocab[ch] for ch in sentence], dtype=torch.long).to(device)
        hidden, cell = encoder(src_seq)
        
        # Start with first target char from dataset (could be a special <SOS>)
        input_decoder = torch.tensor([trg_encoded[0][0]], dtype=torch.long).to(device)
        result = []
        for _ in range(max_len):
            output, hidden, cell = decoder(input_decoder, hidden, cell)
            pred_idx = output.argmax(1).item()
            result.append(list(trg_vocab.keys())[pred_idx])
            input_decoder = torch.tensor([pred_idx], dtype=torch.long).to(device)
        return "".join(result)

# Test translation
for s in src_sentences:
    print("Input:", "".join(s), "-> Output:", translate(s))


# Result

# Epoch 20, Loss: 0.0090
# Epoch 40, Loss: 0.0033
# Epoch 60, Loss: 0.0019
# Epoch 80, Loss: 0.0012
# Epoch 100, Loss: 0.0009
# Epoch 120, Loss: 0.0007
# Epoch 140, Loss: 0.0005
# Epoch 160, Loss: 0.0004
# Epoch 180, Loss: 0.0004
# Epoch 200, Loss: 0.0003