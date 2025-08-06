import streamlit as st
import torch
import torch.nn as nn
import pickle

# ---------------------------------------------
# Define character mappings (same used in training)
chars = ['\n', ' ', '!', "'", ',', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
         'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z']  # Add more characters if needed

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}


with open("vocab.pkl", "rb") as f:
    chars, char2idx, idx2char = pickle.load(f)
vocab_size = len(chars)


# ---------------------------------------------
# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# ---------------------------------------------
# GRU model class
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

# ---------------------------------------------
# Load the model (LSTM or GRU)
@st.cache_resource
def load_model(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "LSTM":
        model = LSTMModel(vocab_size)
        model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
    else:
        model = GRUModel(vocab_size)
        model.load_state_dict(torch.load("gru_model.pth", map_location=device))
    
    model.to(device)
    model.eval()
    return model

# ---------------------------------------------
# Generate text function
def generate_text(model, seed_text, length=300):
    device = next(model.parameters()).device
    model.eval()

    input_seq = torch.tensor([char2idx.get(c, 0) for c in seed_text.lower()], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    result = seed_text

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq[:, -100:], hidden)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            predicted_idx = torch.multinomial(probs, num_samples=1).item()
            result += idx2char[predicted_idx]
            input_seq = torch.cat([input_seq, torch.tensor([[predicted_idx]], device=device)], dim=1)

    return result

# ---------------------------------------------
# Streamlit UI
st.title("🧠 Character-Level Text Generation with PyTorch")
st.markdown("Train a model on your own text and generate new sequences using **LSTM** or **GRU**!")

model_type = st.selectbox("Choose Model Type", ["LSTM", "GRU"])
seed_text = st.text_input("Seed text", value="this is")
length = st.slider("Generate Length", 50, 1000, 300, step=50)

if st.button("Generate"):
    model = load_model(model_type)
    output = generate_text(model, seed_text, length)
    st.subheader("Generated Text:")
    st.write(output)
