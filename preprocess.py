
import pandas as pd
import re
import json
from collections import Counter

# Load the three datasets
train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

# Simple word tokenizer
def tokenize(text):
    return re.findall(
        r"\w+|[^\w\s]",
        str(text).lower()
    )

# Build vocabulary using TRAINING data only
counter = Counter()

for text in pd.concat([
    train["LLM_Input"],
    train["LLM_output"]
]):
    counter.update(tokenize(text))

# Reserved tokens
vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SEP>": 2
}

MAX_VOCAB = 20000

for word, count in counter.most_common(MAX_VOCAB - len(vocab)):
    if word not in vocab:
        vocab[word] = len(vocab)

# Encode a text sequence
MAX_LENGTH = 256

def encode(text, max_length=MAX_LENGTH):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    ids = ids[:max_length]

    ids += [vocab["<PAD>"]] * (
        max_length - len(ids)
    )

    return ids

# Save vocabulary
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

print("Vocabulary size:", len(vocab))

print("\nExample tokenization:")
example = train["LLM_output"].iloc[0]

print("Original:", example[:200])
print("Tokens:", tokenize(example)[:20])
print("Encoded:", encode(example)[:20])

print("\nAll datasets loaded successfully.")
