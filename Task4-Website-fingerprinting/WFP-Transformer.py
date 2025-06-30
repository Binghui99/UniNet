import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
import time
# Set a seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load the dataset
npz_file_path = "C:\\Users\\bingh\\Desktop\\MIL\\CIC-2018-csv\\2024DOQ\\dataset_100_200.npz"
data = np.load(npz_file_path)
x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]

# Create a validation set from the training set
x_train_len = len(x_train)
split_idx = int(x_train_len * 0.75)
x_val, y_val = x_train[split_idx:], y_train[split_idx:]
x_train, y_train = x_train[:split_idx], y_train[:split_idx]

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Normalize the input data to help stabilize Transformer training
mean = x_train.mean(dim=[0,1], keepdim=True)
std = x_train.std(dim=[0,1], keepdim=True) + 1e-7
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std

# Define model parameters
CLASSES = 101
EPOCHS = 1
BATCH_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0001
REGULARIZATION = 0.001

# Create DataLoaders
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Define the Positional Encoding and Transformer-based model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)    # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # add positional encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Mean pooling over sequence
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.fc(x)      # (batch_size, num_classes)
        return x

# Initialize the model
input_size = x_train.shape[2]
seq_len = x_train.shape[1]
model = TransformerClassifier(input_size, CLASSES, seq_len, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=DROPOUT)

# Optional: Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)

# Early stopping parameters
best_val_loss = float('inf')
early_stop_count = 0
PATIENCE = 10

train_acc_history, val_acc_history = [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)
        train_correct += (outputs.argmax(1) == y_batch).sum().item()

    train_loss /= len(x_train)
    train_acc = train_correct / len(x_train)
    train_acc_history.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * x_batch.size(0)
            val_correct += (outputs.argmax(1) == y_batch).sum().item()

    val_loss /= len(x_val)
    val_acc = val_correct / len(x_val)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_count = 0
        torch.save(model.state_dict(), "best_transformer_model.pth")
    else:
        early_stop_count += 1

    if early_stop_count >= PATIENCE:
        print("Early stopping triggered.")
        break

# Load the best model
model.load_state_dict(torch.load("best_transformer_model.pth"))

# Evaluate on test data
model.eval()



# Select a random batch from the test dataset
sample_batch, _ = next(iter(test_loader))
sample_batch = sample_batch.to(device)

# Measure inference time
start_time = time.time()
with torch.no_grad():
    _ = model(sample_batch)
end_time = time.time()

# Calculate the time taken for inference
inference_time = end_time - start_time
print(f"Inference time for a batch of size {sample_batch.size(0)}: {inference_time:.6f} seconds")

# Optionally, calculate per-sample inference time
per_sample_time = inference_time / sample_batch.size(0)
print(f"Average inference time per sample: {per_sample_time:.6f} seconds")


test_correct = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        test_correct += (outputs.argmax(1) == y_batch).sum().item()

test_acc = test_correct / len(x_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training progress
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.title("Training Progress (Transformer)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
