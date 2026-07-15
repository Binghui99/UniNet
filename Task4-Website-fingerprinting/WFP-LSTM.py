import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time


# Load the dataset
npz_file_path = "C:\\Users\\bingh\Desktop\\MIL\\CIC-2018-csv\\2024DOQ\\dataset_100_200.npz"
# npz_file_path = "C:\\Users\\bingh\Desktop\\MIL\\CIC-2018-csv\\2024DOQ\\doq_100_360_47500_4_ow.npz"
data = np.load(npz_file_path)
x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
x_train_len = len(x_train)
x_val, y_val = x_train[int(x_train_len * 0.75):], y_train[int(x_train_len * 0.75):]
x_train, y_train = x_train[:int(x_train_len * 0.75)], y_train[:int(x_train_len * 0.75)]

# Convert to PyTorch tensors
x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Define parameters
CLASSES = 101
EPOCHS = 120
BATCH_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0001
REGULARIZATION = 0.001

# Create DataLoader for batching
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Define the model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout, regularization):
        super(BiLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm4 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_size = x_train.shape[2]
hidden_size = 64
model = BiLSTMModel(input_size, hidden_size, CLASSES, DROPOUT, REGULARIZATION)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
best_val_loss = float('inf')
early_stop_count = 0
PATIENCE = 3

train_acc_history, val_acc_history = [], []
#
# for epoch in range(EPOCHS):
#     model.train()
#     train_loss, train_correct = 0, 0
#     for x_batch, y_batch in train_loader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(x_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         train_correct += (outputs.argmax(1) == y_batch).sum().item()
#
#     train_acc = train_correct / len(x_train)
#     train_acc_history.append(train_acc)
#
#     # Validation
#     model.eval()
#     val_loss, val_correct = 0, 0
#     with torch.no_grad():
#         for x_batch, y_batch in val_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#
#             val_loss += loss.item()
#             val_correct += (outputs.argmax(1) == y_batch).sum().item()
#
#     val_acc = val_correct / len(x_val)
#     val_acc_history.append(val_acc)
#
#     print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         early_stop_count = 0
#         torch.save(model.state_dict(), "best_model.pth")
#     else:
#         early_stop_count += 1
#
#     if early_stop_count >= PATIENCE:
#         print("Early stopping triggered.")
#         break

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

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
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
