
import json
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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

print("Using device:", device, flush=True)

# ==========================================
# 2. LOAD DATA AND VOCABULARY
# ==========================================

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

with open("vocab.json", "r") as f:
    vocab = json.load(f)

print("Training examples:", len(train))
print("Validation examples:", len(val))
print("Testing examples:", len(test))

# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

def tokenize(text):
    return re.findall(
        r"<sep>|\w+|[^\w\s]",
        str(text).lower()
    )


def encode_tokens(tokens):
    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    ids = ids[:MAX_LENGTH]
    length = max(1, len(ids))

    ids += [vocab["<PAD>"]] * (
        MAX_LENGTH - len(ids)
    )

    return ids, length


def encode_row(row, mode):

    prompt = tokenize(row["LLM_Input"])
    response = tokenize(row["LLM_output"])

    if mode == "input":
        tokens = prompt[:MAX_LENGTH]

    elif mode == "both":
        # Reserve space for both prompt and response
        prompt = prompt[:64]
        response = response[:MAX_LENGTH - len(prompt) - 1]

        tokens = prompt + ["<SEP>"] + response

    else:
        tokens = response[:MAX_LENGTH]

    return encode_tokens(tokens)


def prepare_data(df, mode):

    encoded = [
        encode_row(row, mode)
        for _, row in df.iterrows()
    ]

    X = np.array(
        [item[0] for item in encoded],
        dtype=np.int64
    )

    lengths = np.array(
        [item[1] for item in encoded],
        dtype=np.int64
    )

    y = np.array(
        [LABELS[name] for name in df["LLM_name"]],
        dtype=np.int64
    )

    return TensorDataset(
        torch.tensor(X, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(y, dtype=torch.long)
    )

# ==========================================
# 4. CNN MODEL
# ==========================================

class TextCNN(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 128, padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(128, 64, kernel_size=k)
            for k in [3, 4, 5]
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 3, 3)

    def forward(self, x, lengths):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        features = []

        for conv in self.convs:
            c = torch.relu(conv(x))
            pooled = torch.max(c, dim=2).values
            features.append(pooled)

        x = torch.cat(features, dim=1)
        x = self.dropout(x)

        return self.fc(x)

# ==========================================
# 5. LSTM MODEL
# ==========================================

class TextLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 128, padding_idx=0
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

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        features = self.dropout(hidden[-1])

        return self.fc(features)

# ==========================================
# 6. EVALUATE MODEL
# ==========================================

def evaluate(model, loader):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for X, lengths, y in loader:

            X = X.to(device)

            outputs = model(X, lengths)
            preds = outputs.argmax(dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(y.numpy())

    accuracy = accuracy_score(
        actual, predictions
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1, actual, predictions

# ==========================================
# 7. TRAIN ONE EXPERIMENT
# ==========================================

def run_experiment(model_name, mode):

    print("\n" + "=" * 50, flush=True)
    print(
        f"TRAINING {model_name.upper()} | "
        f"MODE: {mode.upper()}",
        flush=True
    )
    print("=" * 50, flush=True)

    # Reset random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader = DataLoader(
        prepare_data(train, mode),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        prepare_data(val, mode),
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        prepare_data(test, mode),
        batch_size=BATCH_SIZE
    )

    if model_name == "CNN":
        model = TextCNN(len(vocab)).to(device)
    else:
        model = TextLSTM(len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1

    checkpoint = (
        f"best_{model_name.lower()}_{mode}.pth"
    )

    # Training loop
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

            if model_name == "LSTM":
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

            optimizer.step()

            total_loss += loss.item() * len(y)

        average_loss = (
            total_loss / len(train_loader.dataset)
        )

        val_acc, _, _, _ = evaluate(
            model, val_loader
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: {val_acc:.4f}",
            flush=True
        )

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            torch.save(
                model.state_dict(),
                checkpoint
            )

    # Load best validation model
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    # Final test evaluation
    test_acc, macro_f1, actual, predictions = (
        evaluate(model, test_loader)
    )

    print("\nFINAL RESULTS")
    print("Model:", model_name)
    print("Mode:", mode)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

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

    return {
        "Model": model_name,
        "Input Type": mode,
        "Test Accuracy": test_acc,
        "Macro F1": macro_f1,
        "Best Validation Accuracy": best_val_accuracy
    }

# ==========================================
# 8. RUN ALL RQ2 EXPERIMENTS
# ==========================================

if __name__ == "__main__":

    results = []

    experiments = [
        ("CNN", "input"),
        ("CNN", "both"),
        ("LSTM", "input"),
        ("LSTM", "both")
    ]

    for model_name, mode in experiments:

        result = run_experiment(
            model_name, mode
        )

        results.append(result)

        # Save after every experiment
        pd.DataFrame(results).to_csv(
            "rq2_results.csv",
            index=False
        )

    # Add previously completed RQ1 results
    results.extend([
        {
            "Model": "CNN",
            "Input Type": "output",
            "Test Accuracy": 0.7237,
            "Macro F1": 0.73,
            "Best Validation Accuracy": 0.7829
        },
        {
            "Model": "LSTM",
            "Input Type": "output",
            "Test Accuracy": 0.5461,
            "Macro F1": 0.55,
            "Best Validation Accuracy": 0.5461
        }
    ])

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "all_results.csv",
        index=False
    )

    print("\n" + "=" * 50)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 50)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved: all_results.csv")
    print("Saved: rq2_results.csv")
