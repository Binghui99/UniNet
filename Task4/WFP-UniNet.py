import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

# Inspired by VIT
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import pandas as pd
import numpy as np
import random
from numpy import load

from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# For metrics and ROC curves
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# ----------------------------
# GLOBAL PARAMETERS
# ----------------------------
open_world = False
parm_num_classes = 300 + 1 + int(open_world)
parm_num_ow_classes = 4500

parm_feature_size = 200   # Original "sequence length"
parm_emb_size = 64
parm_word_size = 5        # or 3, depending on your dataset
parm_num_heads = 32
parm_num_encoders = 1
parm_dense_layer_1 = 1024
parm_dense_layer_2 = 512

N_EPOCHS = 100    # reduce for quick debug; set 120 in real run
LR = 0.01

# ----------------------------
# 1) LOAD .npz DATA
# ----------------------------
npz_file_path = (
    f"C:\\Users\\bingh\\Desktop\\MIL\\CIC-2018-csv\\2024DOQ\\Dataset_Paper\\close-world\\dataset_{parm_num_classes - 1 - int(open_world)}_200.npz"
)
dict_data = load(npz_file_path)
x_train, y_train, x_test, y_test = (
    dict_data["x_train"],
    dict_data["y_train"],
    dict_data["x_test"],
    dict_data["y_test"],
)

print("Initial shapes:")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:", x_test.shape, "y_test:", y_test.shape)

# ----------------------------
# 2) (OPTIONAL) MERGE OPEN-WORLD DATA IF open_world = True
# ----------------------------
if open_world:
    ow_npz_file_path = "C:\\Users\\bingh\\Desktop\\MIL\\CIC-2018-csv\\2024DOQ\\Dataset_Paper\\open_world\\dataset_4500_ow.npz"
    ow_dict_data = load(ow_npz_file_path)
    x_ow = ow_dict_data["x_ow"]
    y_ow = np.array([parm_num_classes - 1] * len(x_ow))  # label for unknown
    np.random.shuffle(x_ow)

    dataset_ow_len = parm_num_ow_classes
    ow_split = int(dataset_ow_len * 0.6)
    x_ow_train, x_ow_test = x_ow[:ow_split], x_ow[ow_split:dataset_ow_len]
    y_ow_train, y_ow_test = y_ow[:ow_split], y_ow[ow_split:dataset_ow_len]

    print("Open-world shapes:")
    print("x_ow_train:", x_ow_train.shape, "x_ow_test:", x_ow_test.shape)
    print("Unique y_ow:", np.unique(y_ow))

    # Merge
    x_train = np.concatenate((x_train, x_ow_train), axis=0)
    y_train = np.concatenate((y_train, y_ow_train), axis=0)
    x_test = np.concatenate((x_test, x_ow_test), axis=0)
    y_test = np.concatenate((y_test, y_ow_test), axis=0)

    print("After merging open-world:")
    print("x_train:", x_train.shape, "y_train:", y_train.shape)
    print("x_test:", x_test.shape, "y_test:", y_test.shape)


# ----------------------------
# 3) FUNCTION TO ADD 4 EXTRA ROWS (MIN, MAX, MEAN, STD)
#    -> (200, D) -> (204, D)
# ----------------------------
def add_session_features(x_array: np.ndarray):
    """
    x_array shape: (N, 200, D)
    returns shape: (N, 200+4, D)
    """
    new_list = []
    for sample in x_array:
        # sample.shape = (200, D)
        row_min = np.min(sample, axis=0, keepdims=True)
        row_max = np.max(sample, axis=0, keepdims=True)
        row_mean = np.mean(sample, axis=0, keepdims=True)
        row_std = np.std(sample, axis=0, keepdims=True)

        # Concatenate these 4 rows
        augmented = np.concatenate([sample, row_min, row_max, row_mean, row_std], axis=0)
        new_list.append(augmented)

    return np.array(new_list)


# ----------------------------
# 4) AUGMENT THE TRAIN/TEST
# ----------------------------
print("\nAugmenting session-level features (adding 4 extra rows per sample)...")
x_train = add_session_features(x_train)  # shape: (N, 204, D)
x_test = add_session_features(x_test)    # shape: (M, 204, D)

print("After augmentation:")
print("x_train:", x_train.shape, "x_test:", x_test.shape)

# Update parm_feature_size to reflect 200 + 4 = 204
parm_feature_size += 4

# ----------------------------
# 5) CONVERT TO TORCH TENSORS
# ----------------------------
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# ------------------------------------------------
# 6) MODEL DEFINITIONS WITH FLAG EMBEDDING
# ------------------------------------------------

class NetPatchEmbedding(nn.Module):
    """
    Now we add a small Embedding to mark each row as
    either packet-level (flag=0) or session-level (flag=1).
    The code assumes the last 4 rows are session-level.
    """
    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = parm_feature_size,  # e.g., 204
        emb_size: int = parm_emb_size,
        word_size: int = parm_word_size
    ):
        super().__init__()
        self.feature_size = feature_size  # 204
        self.emb_size = emb_size

        # Linear projection of each row's D-dim -> emb_size
        self.projection = nn.Linear(word_size, emb_size)

        # SLA token: aggregates sequence-level info
        self.sla_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # Learnable positional embedding: shape (feature_size+1, emb_size)
        self.positions = nn.Parameter(torch.randn(feature_size + 1, emb_size))

        # Embedding for the "flag": 2 possible flags (0=packet, 1=session-level)
        self.flag_embed = nn.Embedding(2, emb_size)  # shape=(2, emb_size)

        # Precompute the flag indices for each row:
        # first 200 = 0, last 4 = 1
        self.register_buffer(
            "row_flags",
            torch.tensor([0]* (feature_size - 4) + [1]*4, dtype=torch.long)
        )
        # row_flags.shape = (204,)

    def forward(self, x: Tensor) -> Tensor:
        """
        x.shape = (B, feature_size, word_size)  => e.g. (B, 204, 5)
        output.shape = (B, feature_size+1, emb_size)
        """
        b, seq_len, _ = x.shape  # e.g., (B, 204, 5)

        # 1) Project input from word_size -> emb_size
        x_lin = self.projection(x)  # => (B, 204, emb_size)

        # 2) SLA token
        sla_tokens = repeat(self.sla_token, "1 n e -> b n e", b=b)
        # after repeat => shape (B, 1, emb_size)

        # 3) Add positional embedding
        #    positions.shape = (205, emb_size)
        #    we slice the first (seq_len+1) for safety
        pos = self.positions[:seq_len+1, :]  # => (seq_len+1, emb_size)
        pos = repeat(pos, "n e -> b n e", b=b)

        # 4) Add the row-level "flag" embedding
        #    row_flags.shape = (seq_len=204,)
        #    -> embed => shape (204, emb_size)
        #    -> repeat => (B, 204, emb_size)
        flag_emb = self.flag_embed(self.row_flags)   # (204, emb_size)
        flag_emb = repeat(flag_emb, "n e -> b n e", b=b)  # => (B, 204, emb_size)

        # 5) Combine everything
        #    Combine x_lin + the corresponding flag_emb
        x_out = x_lin + flag_emb

        # 6) Insert SLA token at the front
        #    So final shape => (B, 205, emb_size)
        x_out = torch.cat([sla_tokens, x_out], dim=1)  # => (B, 1 + seq_len, emb_size)

        # 7) Now add positional embedding
        x_out = x_out + pos

        return x_out


class NetMultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = parm_emb_size, num_heads: int = parm_num_heads, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Split into heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x),"b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # scale and softmax
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class NetResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class NetFeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class NetTransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = parm_emb_size,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
        **kwargs
    ):
        super().__init__(
            NetResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    NetMultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            NetResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    NetFeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class NetTransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = parm_num_encoders, **kwargs):
        super().__init__(*[NetTransformerEncoderBlock(**kwargs) for _ in range(depth)])


class NetClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = parm_emb_size, n_classes: int = parm_num_classes):
        super().__init__(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, parm_dense_layer_1),
            nn.ReLU(),
            nn.Linear(parm_dense_layer_1, parm_dense_layer_2),
            nn.ReLU(),
            nn.Linear(parm_dense_layer_2, n_classes),
        )


class NetFormer(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = parm_feature_size,
        emb_size: int = parm_emb_size,
        depth: int = parm_num_encoders,
        n_classes: int = parm_num_classes,
        **kwargs
    ):
        super().__init__(
            NetPatchEmbedding(in_channels, feature_size, emb_size, parm_word_size),
            NetTransformerEncoder(depth, emb_size=emb_size, **kwargs),
            NetClassificationHead(emb_size, n_classes),
        )


print("\nModel summary:")
print(summary(NetFormer(), (parm_feature_size, parm_word_size), device="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
print(f"Using device: {device} ({device_name})")

# Instantiate model
model = NetFormer().to(device)

# ----------------------------
# 7) TRAINING LOOP (Short Demo)
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in trange(N_EPOCHS, desc="Training"):
    start_time = time.time()
    train_loss = 0.0
    all_preds, all_labels = [], []

    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} in training", leave=False):
        x, y = batch
        y = y.type(torch.LongTensor).to(device)
        x = x.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        # optimizer.step()

        train_loss += loss.detach().cpu().item()
        all_preds.append(torch.argmax(y_hat, dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    # if epoch == 80:
    #     scheduler = StepLR(optimizer, step_size=5, gamma=0.75)
    # scheduler.step()

    avg_train_loss = train_loss / len(train_dataloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # metrics
    train_acc = accuracy_score(all_labels, all_preds) * 100.0
    train_recall = recall_score(all_labels, all_preds, average="macro") * 100.0
    train_precision = precision_score(all_labels, all_preds, average="macro") * 100.0

    end_time = time.time()
    print(f"\n[Epoch {epoch+1}/{N_EPOCHS}] - LR: {optimizer.param_groups[0]['lr']:.5f}")
    print(f"   Training loss: {avg_train_loss:.4f}")
    print(f"   Accuracy:   {train_acc:.2f}%")
    print(f"   Recall:     {train_recall:.2f}%")
    print(f"   Precision:  {train_precision:.2f}%")
    print(f"   (Epoch time: {end_time - start_time:.2f}s)")

# ----------------------------
# 8) SAVE & LOAD MODEL
# ----------------------------
model_save_path = "model_transformer.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}.")

loaded_model = NetFormer().to(device)
loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
loaded_model.eval()
print("Model loaded from disk. Ready for evaluation on test set.")

# ----------------------------
# 9) TEST EVALUATION
# ----------------------------
test_loss = 0.0
all_preds, all_labels = [], []
all_logits = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing", leave=False):
        x, y = batch
        y = y.type(torch.LongTensor).to(device)
        x = x.to(device)

        logits = loaded_model(x)
        loss = criterion(logits, y)
        test_loss += loss.detach().cpu().item()

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())


# 1) Convert lists to arrays
test_loss /= len(test_dataloader)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_logits = np.concatenate(all_logits)

# 2) Compute basic metrics (Loss, Acc, Recall, Precision)
test_acc = accuracy_score(all_labels, all_preds) * 100.0
test_recall = recall_score(all_labels, all_preds, average="macro") * 100.0
test_precision = precision_score(all_labels, all_preds, average="macro") * 100.0

print(f"\n[Test Results]")
print(f"Test Loss:       {test_loss:.4f}")
print(f"Test Accuracy:   {test_acc:.2f}%")
print(f"Test Recall:     {test_recall:.2f}%")
print(f"Test Precision:  {test_precision:.2f}%")

# 3) Compute multi-class probabilities (via softmax)
all_probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()  # shape: (N, num_classes)

# 4) Binarize labels for one-vs-rest analysis
all_labels_bin = label_binarize(all_labels, classes=range(parm_num_classes))
# shape: (N, num_classes)

# 5) Compute micro-average ROC
fpr_micro, tpr_micro, thr_micro = roc_curve(
    all_labels_bin.ravel(),
    all_probs.ravel()
)
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 6) Print TPR/FPR values for each threshold (micro-average)
print("\n=== Micro-average ROC thresholds ===")
for thr_val, fpr_val, tpr_val in zip(thr_micro, fpr_micro, tpr_micro):
    print(f"Threshold={thr_val:.3f}, FPR={fpr_val:.3f}, TPR={tpr_val:.3f}")

# 7) Plot TPR vs. FPR (micro-average only)
plt.figure(figsize=(7, 6))
plt.plot(
    fpr_micro,
    tpr_micro,
    label=f"Micro-average (AUC = {roc_auc_micro:.2f})",
    color="navy",
    linestyle="-",
    linewidth=2,
)
# Plot reference diagonal
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Overall (Micro-average) ROC Curve")
plt.legend(loc="lower right")
plt.show()