
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/lstm_fd001.pt"
DATA_PATH = ROOT / "data/raw/test_FD001.txt"

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


class RUL_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :]).squeeze(-1)


def load_predictor():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = RUL_LSTM(checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint


def predict_engine(engine_data, model, checkpoint, buffer=5):
    sensors = checkpoint["sensor_names"]
    window = checkpoint["window"]

    engine_data = engine_data.sort_values("cycle")

    if len(engine_data) < window:
        raise ValueError(
            f"At least {window} cycles are required."
        )

    # Select the most recent 30 cycles
    recent = engine_data.tail(window)

    cycles = recent["cycle"].to_numpy()

    if not np.all(np.diff(cycles) == 1):
        raise ValueError(
            "The latest 30 cycles must be consecutive."
        )

    # Apply saved training normalization
    values = recent[sensors].to_numpy(dtype=np.float64)

    mean = np.array(checkpoint["scaler_mean"])
    scale = np.array(checkpoint["scaler_scale"])

    values = ((values - mean) / scale).astype(np.float32)

    if not np.isfinite(values).all():
        raise ValueError("Invalid sensor readings detected.")

    # Shape: (1, 30, 15)
    x = torch.tensor(values).unsqueeze(0)

    with torch.no_grad():
        raw_rul = (
            model(x).item() * checkpoint["rul_scale"]
        )

    raw_rul = max(0.0, raw_rul)
    final_rul = max(0.0, raw_rul - buffer)

    return {
        "latest_cycle": int(cycles[-1]),
        "raw_rul": round(raw_rul, 2),
        "predicted_rul": round(final_rul, 2),
        "buffer": buffer
    }


def predict_history(engine_data, model, checkpoint, buffer=5):
    """Predict RUL at every cycle with at least 30 readings."""

    engine_data = engine_data.sort_values("cycle")
    window = checkpoint["window"]
    sensors = checkpoint["sensor_names"]

    if len(engine_data) < window:
        raise ValueError(f"At least {window} cycles required.")

    cycles = engine_data["cycle"].to_numpy()

    if not np.all(np.diff(cycles) == 1):
        raise ValueError("Engine cycles must be consecutive.")

    # Use the original training normalization
    values = engine_data[sensors].to_numpy(dtype=np.float64)

    mean = np.array(checkpoint["scaler_mean"])
    scale = np.array(checkpoint["scaler_scale"])

    values = ((values - mean) / scale).astype(np.float32)

    if not np.isfinite(values).all():
        raise ValueError("Invalid sensor readings.")

    # Create a sliding 30-cycle window for every prediction
    windows = np.stack([
        values[i:i + window]
        for i in range(len(values) - window + 1)
    ])

    # Predict in batches for efficiency
    predictions = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(windows), 128):
            batch = torch.from_numpy(windows[i:i + 128])
            pred = model(batch).cpu().numpy()
            predictions.extend(pred)

    raw_rul = np.maximum(
        0,
        np.array(predictions) * checkpoint["rul_scale"]
    )

    final_rul = np.maximum(0, raw_rul - buffer)

    return pd.DataFrame({
        "cycle": cycles[window - 1:],
        "raw_rul": raw_rul,
        "predicted_rul": final_rul
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-id", type=int, default=1)
    args = parser.parse_args()

    # Load NASA test histories, without using RUL answers
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    engine = df[df["engine_id"] == args.engine_id]

    if engine.empty:
        raise ValueError("Engine ID not found.")

    model, checkpoint = load_predictor()

    result = predict_engine(engine, model, checkpoint)

    print("\n=== PREDICTIVE MAINTENANCE ===")
    print("Engine ID:", args.engine_id)
    print("Latest cycle:", result["latest_cycle"])
    print("Raw predicted RUL:", result["raw_rul"])
    print("Safety buffer:", result["buffer"])
    print("Final estimated RUL:", result["predicted_rul"])
    print("\nExperimental estimate — not a maintenance guarantee.")
