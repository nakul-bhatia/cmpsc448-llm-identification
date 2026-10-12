
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
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
# 2. LOAD DATA
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
    tokens = tokenize(text)[:MAX_LENGTH]

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    length = max(1, len(ids))

    ids += [vocab["<PAD>"]] * (
        MAX_LENGTH - len(ids)
    )

    return ids, length

def prepare_data(df):
    encoded = [
        encode(text)
        for text in df["LLM_output"]
    ]

    X = np.array(
        [item[0] for item in encoded],
        dtype=np.int64
    )

    lengths = np.array(
        [item[1] for item in encoded],
        dtype=np.int64
    )

    y = np.array([
        LABELS[name]
        for name in df["LLM_name"]
    ], dtype=np.int64)

    return TensorDataset(
        torch.tensor(X, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
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
# 3. DEFINE LSTM
# ==========================================

class TextLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            128,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 3)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Ignore padding during sequence processing
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Final hidden state of each sequence
        features = hidden[-1]

        features = self.dropout(features)

        return self.fc(features)

model = TextLSTM(len(vocab)).to(device)

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
        for X, lengths, y in loader:
            X = X.to(device)
            lengths = lengths.to(device)

            outputs = model(X, lengths)
            preds = outputs.argmax(dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )
            actual.extend(y.numpy())

    accuracy = accuracy_score(
        actual, predictions
    )

    return accuracy, actual, predictions

# ==========================================
# 5. TRAIN LSTM
# ==========================================

best_val_accuracy = -1

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, lengths, y in train_loader:
        X = X.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X, lengths)
        loss = criterion(outputs, y)

        loss.backward()

        # Reduce exploding gradient risk
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(y)

    average_loss = (
        total_loss / len(train_loader.dataset)
    )

    val_acc, _, _ = evaluate(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}",
        flush=True
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc

        torch.save(
            model.state_dict(),
            "best_lstm.pth"
        )

# ==========================================
# 6. FINAL TEST
# ==========================================

model.load_state_dict(
    torch.load(
        "best_lstm.pth",
        map_location=device,
        weights_only=True
    )
)

test_acc, actual, predictions = evaluate(
    test_loader
)

print("\nFINAL LSTM TEST RESULTS")
print("-----------------------")
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
