from datasets import load_dataset
import pandas as pd

DATASET_NAME = "lmarena-ai/arena-human-preference-55k"

# Download the dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Convert to pandas DataFrame
df = dataset.to_pandas()

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 model comparisons:")
print(df[["model_a", "model_b"]].head())

print("\nMost common models:")
print(pd.concat([df["model_a"], df["model_b"]])
      .value_counts().head(20))


import json

def get_family(model):
    model = str(model).lower()

    if model.startswith(("gpt-", "chatgpt-", "openai/")):
        return "GPT"
    elif model.startswith("claude-"):
        return "Claude"
    elif model.startswith(("gemini-", "gemini-pro")):
        return "Gemini"

    return None


def parse_messages(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return []


records = []

for _, row in df.iterrows():
    prompts = parse_messages(row["prompt"])

    # Keep only single-turn conversations
    if len(prompts) != 1:
        continue

    prompt = prompts[0]

    if not isinstance(prompt, str) or not prompt.strip():
        continue

    for side in ["a", "b"]:
        model = row[f"model_{side}"]
        family = get_family(model)
        responses = parse_messages(row[f"response_{side}"])

        if family is None or len(responses) != 1:
            continue

        response = responses[0]

        if not isinstance(response, str) or not response.strip():
            continue

        records.append({
            "conversation_id": str(row["id"]),
            "LLM_name": family,
            "model_version": str(model),
            "LLM_Input": prompt.strip(),
            "LLM_output": response.strip()
        })

clean_df = pd.DataFrame(records)

print("Extracted examples:", len(clean_df))
print(clean_df["LLM_name"].value_counts())

# Remove duplicate examples
clean_df = clean_df.drop_duplicates(
    subset=["LLM_name", "LLM_Input", "LLM_output"]
)

# Exclude extremely short responses and prompts
clean_df = clean_df[
    (clean_df["LLM_output"].str.split().str.len() >= 30) &
    (clean_df["LLM_Input"].str.split().str.len() >= 3)
].copy()

# Verify that all three families are present
families = ["GPT", "Claude", "Gemini"]

counts = clean_df["LLM_name"].value_counts()

print("Available examples after filtering:")
print(counts)

if not all(counts.get(name, 0) >= 500 for name in families):
    raise ValueError(
        "Fewer than 500 examples for at least one family. "
        "Reduce the target or inspect the filtering."
    )

# Balanced sample: 500 per family
balanced_df = pd.concat([
    clean_df[clean_df["LLM_name"] == family]
    .sample(n=500, random_state=42)
    for family in families
], ignore_index=True)

# Shuffle rows
balanced_df = balanced_df.sample(
    frac=1, random_state=42
).reset_index(drop=True)

# Save all useful metadata
balanced_df.to_csv(
    "llm_dataset.csv",
    index=False
)

print("\nFinal dataset:")
print(balanced_df["LLM_name"].value_counts())

print("\nSaved llm_dataset.csv")

print(balanced_df.shape)
print(balanced_df.isnull().sum())

print("\nExample:")
example = balanced_df.iloc[0]

print("Family:", example["LLM_name"])
print("Original model:", example["model_version"])
print("Prompt:", example["LLM_Input"])
print("Response:", example["LLM_output"][:500])

