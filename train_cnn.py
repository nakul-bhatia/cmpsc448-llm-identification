
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. SETTINGS
# ==========================================

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

LABELS = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# ==========================================
# 2. LOAD DATA AND VOCABULARY
# ==========================================

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

with open("vocab.json", "r") as f:
    vocab = json.load(f)

def tokenize(text):
    return re.findall(
        r"\w+|[^\w\s]",
        str(text).lower()
    )

def encode(text):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ][:MAX_LENGTH]

    ids += [vocab["<PAD>"]] * (
        MAX_LENGTH - len(ids)
    )

    return ids

# RQ1: Output text only
def prepare_data(df):
    X = np.array([
        encode(text)
        for text in df["LLM_output"]
    ], dtype=np.int64)

    y = np.array([
        LABELS[name]
        for name in df["LLM_name"]
    ], dtype=np.int64)

    return TensorDataset(
        torch.tensor(X, dtype=torch.long),
        torch.tensor(y, dtype=torch.long)
    )

train_loader = DataLoader(
    prepare_data(train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    prepare_data(val),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    prepare_data(test),
    batch_size=BATCH_SIZE
)

# ==========================================
# 3. DEFINE CNN MODEL
# ==========================================

class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            128,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(128, 64, kernel_size=k)
            for k in [3, 4, 5]
        ])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(64 * 3, 3)

    def forward(self, x):
        # Input: (batch, sequence_length)
        x = self.embedding(x)

        # (batch, embedding_dim, sequence_length)
        x = x.permute(0, 2, 1)

        features = []

        for conv in self.convs:
            c = torch.relu(conv(x))
            pooled = torch.max(c, dim=2).values
            features.append(pooled)

        x = torch.cat(features, dim=1)
        x = self.dropout(x)

        return self.fc(x)

model = TextCNN(len(vocab)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate(loader):
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )
            actual.extend(y.numpy())

    accuracy = accuracy_score(
        actual, predictions
    )

    return accuracy, actual, predictions

# ==========================================
# 5. TRAIN MODEL
# ==========================================

best_val_accuracy = -1

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    average_loss = total_loss / len(train_loader.dataset)

    val_acc, _, _ = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}"
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(
            model.state_dict(),
            "best_cnn.pth"
        )

# ==========================================
# 6. TEST BEST MODEL
# ==========================================

model.load_state_dict(
    torch.load(
        "best_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

test_acc, actual, predictions = evaluate(
    test_loader
)

print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Test Accuracy: {test_acc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predictions,
        labels=[0, 1, 2],
        target_names=["GPT", "Claude", "Gemini"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual,
        predictions,
        labels=[0, 1, 2]
    )
)
