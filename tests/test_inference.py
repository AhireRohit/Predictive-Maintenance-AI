
import numpy as np
import pandas as pd
import pytest

from src.predict import (
    load_predictor,
    predict_engine,
    predict_history,
    COLUMNS,
    DATA_PATH,
)
from src.fleet import predict_fleet


@pytest.fixture(scope="module")
def setup():
    model, checkpoint = load_predictor()

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    return model, checkpoint, df


def test_model_loads(setup):
    model, checkpoint, df = setup

    assert checkpoint["input_size"] == 15
    assert checkpoint["window"] == 30
    assert df["engine_id"].nunique() == 100


def test_engine_25_prediction(setup):
    model, checkpoint, df = setup

    engine = df[df["engine_id"] == 25]

    result = predict_engine(
        engine, model, checkpoint
    )

    assert result["latest_cycle"] == 48
    assert abs(result["predicted_rul"] - 135.81) < 0.2


def test_prediction_history(setup):
    model, checkpoint, df = setup

    engine = df[df["engine_id"] == 25]

    history = predict_history(
        engine, model, checkpoint
    )

    # 48 cycles - 29 initial cycles = 19 predictions
    assert len(history) == 19
    assert history["cycle"].iloc[0] == 30
    assert history["cycle"].iloc[-1] == 48

    assert np.isfinite(
        history["predicted_rul"]
    ).all()

    assert (history["predicted_rul"] >= 0).all()


def test_fleet_prediction(setup):
    model, checkpoint, df = setup

    fleet = predict_fleet(
        df, model, checkpoint
    )

    assert len(fleet) == 100
    assert fleet["engine_id"].nunique() == 100
    assert (fleet["predicted_rul"] >= 0).all()

    # Verify sorted order
    assert fleet["predicted_rul"].is_monotonic_increasing

    # Ensure fleet prediction agrees with individual prediction
    engine_25 = fleet[fleet["engine_id"] == 25]

    assert abs(
        engine_25["predicted_rul"].iloc[0] - 135.81
    ) < 0.2


def test_insufficient_history(setup):
    model, checkpoint, df = setup

    engine = df[df["engine_id"] == 25].head(20)

    with pytest.raises(ValueError):
        predict_engine(engine, model, checkpoint)


def test_missing_cycle(setup):
    model, checkpoint, df = setup

    engine = df[df["engine_id"] == 25].copy()

    # Remove cycle 40 from the last 30-cycle history
    engine = engine[engine["cycle"] != 40]

    with pytest.raises(ValueError):
        predict_history(engine, model, checkpoint)
