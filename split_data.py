
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load prepared data
df = pd.read_csv("llm_dataset.csv")

# Normalize prompts to identify duplicates
df["prompt_group"] = (
    df["LLM_Input"]
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

# 80% training, 20% temporary
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)

train_idx, temp_idx = next(
    split1.split(df, groups=df["prompt_group"])
)

train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

# Divide remaining data into validation and test
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)

val_idx, test_idx = next(
    split2.split(temp, groups=temp["prompt_group"])
)

val = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

# Remove grouping helper before saving
for part in [train, val, test]:
    part.drop(columns=["prompt_group"], inplace=True)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("Training:", len(train))
print("Validation:", len(val))
print("Testing:", len(test))

print("\nTraining labels:")
print(train["LLM_name"].value_counts())

print("\nValidation labels:")
print(val["LLM_name"].value_counts())

print("\nTesting labels:")
print(test["LLM_name"].value_counts())
