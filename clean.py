import pandas as pd

# 1. Load the labeled data
df = pd.read_csv("reviews_to_label.csv", encoding="utf-8-sig")
print("Rows loaded:", len(df))
print("\nLabel counts BEFORE cleaning:")
print(df["label"].value_counts(dropna=False))

# 2. Drop rows where the text or the label is missing
df = df.dropna(subset=["text", "label"])

# 3. Tidy up: remove extra spaces, make labels lowercase
df["text"] = df["text"].str.strip()
df["label"] = df["label"].str.strip().str.lower()

# 4. Keep only the two labels we defined
df = df[df["label"].isin(["positive", "negative", "neutral"])]

# 5. Remove duplicate texts
df = df.drop_duplicates(subset="text")

# 6. Report and save
print("\nRows after cleaning:", len(df))
print("\nLabel counts AFTER cleaning:")
print(df["label"].value_counts())

df.to_csv("reviews_clean.csv", index=False)
print("\nSaved reviews_clean.csv")