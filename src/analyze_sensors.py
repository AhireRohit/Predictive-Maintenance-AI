
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load labeled training data
df = pd.read_csv("data/processed/train_FD001_labeled.csv")

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

# 1. Examine sensor variability
print("=== SENSOR STATISTICS ===")

stats = df[sensor_cols].agg(["min", "max", "std"]).T
stats["unique_values"] = df[sensor_cols].nunique()

print(stats.round(4).to_string())

# 2. Identify constant sensors
constant_sensors = stats[
    stats["unique_values"] == 1
].index.tolist()

print("\n=== CONSTANT SENSORS ===")
print(constant_sensors)

# 3. Visualize one engine's sensor history
engine_id = 1
engine = df[df["engine_id"] == engine_id].sort_values("cycle")

selected_sensors = [
    "sensor_2",
    "sensor_4",
    "sensor_11",
    "sensor_15",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, sensor in zip(axes.flat, selected_sensors):
    ax.plot(engine["cycle"], engine[sensor])
    ax.set_title(sensor)
    ax.set_xlabel("Operating Cycle")
    ax.set_ylabel("Sensor Reading")
    ax.grid(alpha=0.3)

fig.suptitle(
    f"Engine {engine_id} - Sensor Degradation Trends",
    fontsize=15
)

plt.tight_layout()

# Save visualization
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "engine_1_sensor_trends.png"

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nVisualization saved to: {output_path}")
