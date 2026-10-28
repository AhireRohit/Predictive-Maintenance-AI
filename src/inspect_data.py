
import pandas as pd

# Column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load the NASA dataset
train = pd.read_csv("data/raw/train_FD001.txt", sep=r"\s+", header=None, names=columns)
test = pd.read_csv("data/raw/test_FD001.txt", sep=r"\s+", header=None, names=columns)
rul = pd.read_csv("data/raw/RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

print("=== DATASET SHAPES ===")
print("Training:", train.shape)
print("Testing:", test.shape)
print("RUL labels:", rul.shape)

print("\n=== NUMBER OF ENGINES ===")
print("Training engines:", train["engine_id"].nunique())
print("Testing engines:", test["engine_id"].nunique())

print("\n=== FIRST 5 ROWS ===")
print(train.head().to_string(index=False))

print("\n=== MISSING VALUES ===")
print("Training:", train.isnull().sum().sum())
print("Testing:", test.isnull().sum().sum())

print("\n=== ENGINE LIFESPANS ===")
print(train.groupby("engine_id")["cycle"].max().describe())
