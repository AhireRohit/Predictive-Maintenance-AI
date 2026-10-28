
from pathlib import Path
import pandas as pd

# Paths
RAW_PATH = Path("data/raw/train_FD001.txt")
OUTPUT_PATH = Path("data/processed/train_FD001_labeled.csv")

# Define column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
df = pd.read_csv(
    RAW_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

# Find the final cycle for each engine
df["max_cycle"] = df.groupby("engine_id")["cycle"].transform("max")

# Calculate Remaining Useful Life
df["RUL"] = df["max_cycle"] - df["cycle"]

# Remove the helper column to avoid leaking the failure endpoint
df = df.drop(columns=["max_cycle"])

# Verify labels
assert (df["RUL"] >= 0).all()
assert (df.groupby("engine_id")["RUL"].last() == 0).all()

# Save processed data
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("=== LABEL CREATION COMPLETE ===")
print("Dataset shape:", df.shape)
print("Total engines:", df["engine_id"].nunique())

print("\n=== ENGINE 1: FIRST 5 CYCLES ===")
print(df[df["engine_id"] == 1][["engine_id", "cycle", "RUL"]].head())

print("\n=== ENGINE 1: LAST 5 CYCLES ===")
print(df[df["engine_id"] == 1][["engine_id", "cycle", "RUL"]].tail())

print("\n=== RUL STATISTICS ===")
print(df["RUL"].describe())

print("\nSaved to:", OUTPUT_PATH)
