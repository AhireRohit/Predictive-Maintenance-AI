
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
from shutil import copyfileobj

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARCHIVE = ROOT / "data" / "CMAPSSData.zip"

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

WINDOW = 30


def prepare_raw_data():
    RAW.mkdir(parents=True, exist_ok=True)

    if all((RAW / name).exists() for name in FILES):
        print("[OK] Raw FD001 files already exist.")
        return

    if not ARCHIVE.exists():
        print("Downloading NASA C-MAPSS dataset...")

        request = Request(
            URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=120) as response:
            with open(ARCHIVE, "wb") as output:
                copyfileobj(response, output)

        print("[OK] Dataset downloaded.")

    with ZipFile(ARCHIVE) as archive:
        for filename in FILES:
            matches = [
                name for name in archive.namelist()
                if Path(name).name == filename
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one archive entry for {filename}, "
                    f"found {len(matches)}"
                )

            with archive.open(matches[0]) as source:
                with open(RAW / filename, "wb") as output:
                    copyfileobj(source, output)

    print("[OK] FD001 files extracted.")


def load_raw(filename):
    return pd.read_csv(
        RAW / filename,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )


def create_labels():
    train = load_raw("train_FD001.txt")

    max_cycles = train.groupby("engine_id")[
        "cycle"
    ].transform("max")

    train["RUL"] = max_cycles - train["cycle"]

    assert len(train) == 20631
    assert train["engine_id"].nunique() == 100
    assert (train["RUL"] >= 0).all()

    train.to_csv(
        PROCESSED / "train_FD001_labeled.csv",
        index=False
    )

    print("[OK] Training RUL labels created.")
    return train


def split_engines(train):
    engine_ids = sorted(train["engine_id"].unique())

    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=0.20,
        random_state=42
    )

    train_df = train[
        train["engine_id"].isin(train_ids)
    ].copy()

    val_df = train[
        train["engine_id"].isin(val_ids)
    ].copy()

    assert not (set(train_ids) & set(val_ids))
    assert len(train_df) == 16561
    assert len(val_df) == 4070

    train_df.to_csv(
        PROCESSED / "train_split.csv",
        index=False
    )

    val_df.to_csv(
        PROCESSED / "val_split.csv",
        index=False
    )

    print("[OK] Engine-level train/validation split created.")
    return train_df, val_df


def engineer_features(data, sensors):
    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    grouped = data.groupby("engine_id")

    features = data[
        ["engine_id", "cycle", "RUL"]
    ].copy()

    for sensor in sensors:
        group = grouped[sensor]

        features[f"{sensor}_latest"] = data[sensor]

        features[f"{sensor}_mean30"] = group.transform(
            lambda x: x.rolling(
                WINDOW, min_periods=WINDOW
            ).mean()
        )

        features[f"{sensor}_std30"] = group.transform(
            lambda x: x.rolling(
                WINDOW, min_periods=WINDOW
            ).std()
        )

        features[f"{sensor}_delta30"] = (
            data[sensor] - group.shift(WINDOW - 1)
        )

    return features.dropna().reset_index(drop=True)


def prepare_features(train_df, val_df):
    all_sensors = [
        f"sensor_{i}" for i in range(1, 22)
    ]

    # Select sensors using training data only
    selected = [
        sensor for sensor in all_sensors
        if train_df[sensor].nunique() > 1
    ]

    train_features = engineer_features(
        train_df, selected
    )

    val_features = engineer_features(
        val_df, selected
    )

    assert train_features.shape == (14241, 63)
    assert val_features.shape == (3490, 63)

    train_features.to_csv(
        PROCESSED / "train_features.csv",
        index=False
    )

    val_features.to_csv(
        PROCESSED / "val_features.csv",
        index=False
    )

    print("[OK] Feature engineering complete.")
    print("Selected sensors:", len(selected))
    print("Training features:", train_features.shape)
    print("Validation features:", val_features.shape)


def prepare_test_data():
    test = load_raw("test_FD001.txt")

    rul = pd.read_csv(
        RAW / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
        names=["true_RUL"]
    )

    endpoints = (
        test.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    assert len(test) == 13096
    assert len(endpoints) == len(rul) == 100
    assert np.array_equal(
        endpoints["engine_id"].to_numpy(),
        np.arange(1, 101)
    )

    endpoints["true_RUL"] = rul["true_RUL"].to_numpy()

    endpoints[
        ["engine_id", "cycle", "true_RUL"]
    ].to_csv(
        PROCESSED / "test_endpoints.csv",
        index=False
    )

    test[test["engine_id"] == 25].to_csv(
        PROCESSED / "engine_25_demo.csv",
        index=False
    )

    print("[OK] Official test endpoints prepared.")
    print("[OK] Demo CSV created.")


def main():
    print("=== NASA C-MAPSS FD001 SETUP ===")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    prepare_raw_data()

    train = create_labels()

    train_df, val_df = split_engines(train)

    prepare_features(train_df, val_df)

    prepare_test_data()

    print("\n=== DATASET SETUP COMPLETE ===")
    print("All FD001 datasets are ready!")


if __name__ == "__main__":
    main()
