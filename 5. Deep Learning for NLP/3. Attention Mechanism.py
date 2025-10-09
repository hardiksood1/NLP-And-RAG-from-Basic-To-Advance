import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1️⃣ Attention Mechanism
# -----------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_dim]
        # encoder_outputs: [seq_len, batch, hidden_dim]
        seq_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(seq_len, 1, 1)  # [seq_len, batch, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [seq_len, batch, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [seq_len, batch]
        return F.softmax(attention, dim=0)  # softmax over seq_len

# -----------------------------
# 2️⃣ Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        
    def forward(self, src):
        # src: [seq_len, batch]
        embedded = self.embedding(src)  # [seq_len, batch, hidden_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [seq_len, batch, hidden_dim]
        return outputs, hidden

# -----------------------------
# 3️⃣ Decoder with Attention
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim + hidden_dim, hidden_dim)  # input + context
        self.fc_out = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch]
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.embedding(input)  # [1, batch, hidden_dim]
        
        # Compute attention weights
        attn_weights = self.attention(hidden, encoder_outputs)  # [seq_len, batch]
        # Weighted sum of encoder outputs
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=0, keepdim=True)  # [1, batch, hidden_dim]
        
        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch, hidden_dim*2]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [1, batch, hidden_dim]
        
        prediction = self.fc_out(torch.cat((output, context), dim=2))  # [1, batch, output_dim]
        return prediction.squeeze(0), hidden, attn_weights

# -----------------------------
# 4️⃣ Seq2Seq Model
# -----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [seq_len, batch], trg: [trg_len, batch]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]  # first token (usually <sos>)
        
        for t in range(1, trg_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[t] if teacher_force else output.argmax(1)
            
        return outputs

# -----------------------------
# 5️⃣ Example Usage
# -----------------------------
INPUT_DIM = 10   # vocab size
OUTPUT_DIM = 10  # vocab size
HIDDEN_DIM = 32
SEQ_LEN = 5
BATCH_SIZE = 2

attn = Attention(HIDDEN_DIM)
enc = Encoder(INPUT_DIM, HIDDEN_DIM)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, attn)
model = Seq2Seq(enc, dec)

# Random data for testing
src = torch.randint(0, INPUT_DIM, (SEQ_LEN, BATCH_SIZE))
trg = torch.randint(0, OUTPUT_DIM, (SEQ_LEN, BATCH_SIZE))

outputs = model(src, trg)  # [trg_len, batch, output_dim]
print("Output shape:", outputs.shape)


# Result
# Output shape: torch.Size([5, 2, 10])
