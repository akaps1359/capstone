import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import glob
import pickle
import os
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
SEQ_LEN = 30  # 15FPS * 2 seconds window (Responsiveness vs Stability)
HIDDEN_DIM = 16
NUM_LAYERS = 1
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_DIR = "data/models"

# --- MODEL DEFINITION ---
class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(GRUAutoencoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, input_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # Encoder: x -> (batch, seq, input) -> hidden state
        _, hidden = self.encoder(x) # hidden: (layers, batch, hidden)
        
        # Decoder: We want to reconstruct x. 
        # For simple AE, we can repeat hidden state or run decoder step-by-step.
        # Here we use a simplified approach: repeating the context vector
        # (A logic often used in simple TS-Reconstruction)
        
        # Expand hidden to match sequence length for decoding input? 
        # Actually, standard way is:
        # Decoder input is usually zeros or previous step output. 
        # But for 'Reconstruction', mapping Embedding->Sequence directly 
        # usually needs a Linear layer or repeating.
        
        # Let's use a Linear layer to map hidden back to sequence sized tensor then reshape?
        # Or simpler: Use Decoder GRU. Input to decoder is 0, hidden is from encoder.
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Decoder input: initialize with zeros
        # Shape: (batch, seq_len, input_dim) - though typical seq2seq uses 1 step at a time.
        # We will feed zeros and see if it can reconstruct the pattern from hidden state.
        decoder_input = torch.zeros_like(x) 
        
        # Run Decoder
        output, _ = self.decoder(decoder_input, hidden)
        
        # Output is (batch, seq, input) - assumes input_dim match. 
        # Since decoder transforms hidden(16) -> input(2), we need a projection layer?
        # Wait, nn.GRU(hidden, input) means input_size=hidden... that's wrong.
        # Decoder GRU should accept input_dim or something and output hidden_dim, then project to input_dim.
        
        # To make it simple and working:
        # Encoder: (B, S, 2) -> (1, B, H)
        # Decoder Input: Repeat (1, B, H) -> (B, S, H)
        # Projection: (B, S, H) -> (B, S, 2)
        
        return output

class SimpleGRUAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGRUAE, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x: (B, S, F)
        _, hidden = self.encoder(x) # hidden: (1, B, H)
        
        # Context vector (last hidden state)
        context = hidden[-1] # (B, H)
        
        # Repeat context for each time step to reconstruct
        # (B, H) -> (B, S, H)
        seq_len = x.size(1)
        context_repeated = context.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Project back to feature space
        # (B, S, H) -> (B, S, F)
        reconstruction = self.decoder_fc(context_repeated)
        return reconstruction

def create_sequences(data, seq_len):
    xs = []
    # Sliding window
    for i in range(len(data) - seq_len):
        x = data[i : i+seq_len]
        xs.append(x)
    return np.array(xs)

def main():
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # 1. Load Data
    files = glob.glob("data/features/*.csv")
    if not files:
        print("❌ No CSV files found in data/features/")
        return
    
    print(f"Loading {len(files)} csv files...")
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df_list.append(df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 2. Preprocessing
    # Features to use: EAR, Pitch
    # Handle NaN (Forward fill then backward fill)
    full_df = full_df[['ear', 'pitch']].ffill().bfill()
    
    raw_data = full_df.values.astype(np.float32)
    
    print(f"Total frames: {len(raw_data)}")
    
    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)
    
    # Create Sequences
    X = create_sequences(scaled_data, SEQ_LEN)
    if len(X) == 0:
        print("Not enough data for sequence length", SEQ_LEN)
        return

    print(f"Training sequences: {X.shape}") # (N, SEQ_LEN, 2)
    
    # PyTorch Dataset
    dataset = TensorDataset(torch.from_numpy(X))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model Setup
    model = SimpleGRUAE(input_dim=2, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training
    print(">>> Start Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for [batch_x] in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.6f}")

    # 5. Threshold Calculation (Evaluation)
    model.eval()
    with torch.no_grad():
        test_input = torch.from_numpy(X)
        reconstruction = model(test_input)
        # MSE per sequence
        # loss dimension: (N, S, F)
        loss_matrix = (test_input - reconstruction) ** 2
        # Mean over features and time -> per sample score
        losses = torch.mean(loss_matrix, dim=[1, 2]).numpy()
        
    # Calculate stats
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    
    # Rule of thumb: Mean + 3*STD or 99th percentile
    threshold = mean_loss + 3 * std_loss
    
    print(f"\n>>> Training Complete.")
    print(f"Loss Mean: {mean_loss:.6f}, Std: {std_loss:.6f}")
    print(f"Calculated Threshold: {threshold:.6f}")
    
    # 6. Save Artifacts
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "gru_ae.pth"))
    
    with open(os.path.join(MODEL_SAVE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(MODEL_SAVE_DIR, "threshold.txt"), "w") as f:
        f.write(str(threshold))
        
    print(f"✅ Model saved to {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()
