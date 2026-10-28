
import numpy as np
import pandas as pd
import torch


def predict_fleet(data, model, checkpoint, buffer=5):
    sensors = checkpoint["sensor_names"]
    window = checkpoint["window"]

    mean = np.array(checkpoint["scaler_mean"])
    scale = np.array(checkpoint["scaler_scale"])

    sequences = []
    records = []

    for engine_id, engine in data.groupby("engine_id"):
        engine = engine.sort_values("cycle")

        if len(engine) < window:
            raise ValueError(
                f"Engine {engine_id} needs at least {window} cycles."
            )

        recent = engine.tail(window)
        cycles = recent["cycle"].to_numpy()

        if not np.all(np.diff(cycles) == 1):
            raise ValueError(
                f"Engine {engine_id} has missing recent cycles."
            )

        values = recent[sensors].to_numpy(dtype=np.float64)
        values = ((values - mean) / scale).astype(np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid sensor values for Engine {engine_id}."
            )

        sequences.append(values)

        records.append({
            "engine_id": int(engine_id),
            "latest_cycle": int(cycles[-1]),
            "sensor_records": len(engine)
        })

    # Batch predictions
    X = torch.from_numpy(np.stack(sequences))

    predictions = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), 64):
            batch = X[start:start + 64]
            output = model(batch).cpu().numpy()
            predictions.extend(output)

    raw_rul = np.maximum(
        0,
        np.array(predictions) * checkpoint["rul_scale"]
    )

    adjusted_rul = np.maximum(0, raw_rul - buffer)

    results = pd.DataFrame(records)
    results["predicted_rul"] = adjusted_rul

    # Engines with the lowest estimated RUL come first
    results = results.sort_values(
        "predicted_rul"
    ).reset_index(drop=True)

    return results
