
from pathlib import Path
from io import BytesIO
from scripts.setup_data import prepare_raw_data
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import (
    load_predictor,
    predict_history,
    COLUMNS,
    DATA_PATH,
)
from src.fleet import predict_fleet


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# DATA & MODEL LOADING
# ==========================================

@st.cache_resource
def get_model():
    return load_predictor()



@st.cache_data(show_spinner="Preparing NASA sample data...")
def load_sample_data():
    # On cloud deployment, raw data will not exist initially.
    # Download and extract it automatically when required.
    if not DATA_PATH.is_file():
        prepare_raw_data()

    return pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )


@st.cache_data
def validate_uploaded_csv(file_bytes):
    data = pd.read_csv(BytesIO(file_bytes))
    data.columns = data.columns.str.strip()

    missing = [c for c in COLUMNS if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data[COLUMNS].copy()

    for col in COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.isna().any().any():
        raise ValueError("Missing or non-numeric values found.")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Infinite sensor values found.")

    for col in ["engine_id", "cycle"]:
        if ((data[col] <= 0) | (data[col] % 1 != 0)).any():
            raise ValueError(f"{col} must contain positive integers.")
        data[col] = data[col].astype(int)

    if data.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle combinations.")

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    window = 30

    for engine_id, engine in data.groupby("engine_id"):
        if len(engine) < window:
            raise ValueError(
                f"Engine {engine_id}: fewer than {window} cycles."
            )

        cycles = engine["cycle"].to_numpy()

        if not np.all(np.diff(cycles) == 1):
            raise ValueError(
                f"Engine {engine_id}: non-consecutive cycles."
            )

    return data


@st.cache_data(show_spinner="Calculating prediction history...")
def get_prediction_history(engine_data):
    model, checkpoint = get_model()
    return predict_history(engine_data, model, checkpoint, buffer=5)


@st.cache_data(show_spinner="Analyzing fleet...")
def get_fleet_predictions(data):
    model, checkpoint = get_model()
    return predict_fleet(data, model, checkpoint, buffer=5)


@st.cache_data
def load_evaluation():
    directory = ROOT / "data" / "processed"

    xgb = pd.read_csv(
        directory / "xgboost_official_predictions.csv"
    )
    lstm = pd.read_csv(
        directory / "lstm_official_predictions.csv"
    )

    xgb = xgb[
        ["engine_id", "true_RUL", "predicted_RUL"]
    ].rename(columns={"predicted_RUL": "XGBoost"})

    lstm = lstm[
        [
            "engine_id",
            "true_RUL",
            "predicted_RUL",
            "predicted_RUL_buffer5",
        ]
    ].rename(columns={
        "predicted_RUL": "LSTM",
        "predicted_RUL_buffer5": "LSTM + 5",
    })

    combined = xgb.merge(
        lstm,
        on=["engine_id", "true_RUL"],
        validate="one_to_one",
    )

    assert len(combined) == 100
    return combined


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("⚙️ Maintenance AI")
st.sidebar.caption("NASA C-MAPSS FD001")

page = st.sidebar.radio(
    "Navigation",
    [
        "Engine Monitor",
        "Fleet Analytics",
        "Model Evaluation",
    ],
)

st.sidebar.divider()

model, checkpoint = get_model()

if page != "Model Evaluation":

    source = st.sidebar.radio(
        "Data Source",
        ["NASA Sample Engines", "Upload CSV"],
    )

    if source == "NASA Sample Engines":
        df = load_sample_data()

    else:
        uploaded = st.sidebar.file_uploader(
            "Upload Engine Sensor CSV",
            type=["csv"],
        )

        if uploaded is None:
            st.info("Upload a compatible CSV to continue.")
            st.stop()

        try:
            df = validate_uploaded_csv(uploaded.getvalue())
            st.sidebar.success("CSV validated successfully!")

        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))
            st.stop()

    threshold = st.sidebar.slider(
        "Maintenance Warning Threshold",
        min_value=5,
        max_value=100,
        value=40,
        step=5,
    )

    st.sidebar.caption(
        "Demonstration threshold, not an approved "
        "maintenance policy."
    )


# ==========================================
# PAGE 1 — ENGINE MONITOR
# ==========================================

if page == "Engine Monitor":

    st.title("Engine Monitoring")
    st.caption(
        "Individual engine health estimates and historical replay"
    )

    engine_ids = sorted(df["engine_id"].unique().tolist())

    engine_id = st.sidebar.selectbox(
        "Select Engine",
        engine_ids,
    )

    full_engine = (
        df[df["engine_id"] == engine_id]
        .sort_values("cycle")
        .copy()
    )

    window = int(checkpoint["window"])
    min_cycle = int(full_engine.iloc[window - 1]["cycle"])
    max_cycle = int(full_engine["cycle"].max())

    st.sidebar.subheader("Historical Replay")

    if min_cycle == max_cycle:
        selected_cycle = max_cycle
        st.sidebar.info(f"Only cycle {max_cycle} is available.")
    else:
        selected_cycle = st.sidebar.slider(
            "Operating Cycle",
            min_value=min_cycle,
            max_value=max_cycle,
            value=max_cycle,
            key=f"replay_{source}_{engine_id}_{min_cycle}_{max_cycle}",
        )

    engine = full_engine[
        full_engine["cycle"] <= selected_cycle
    ].copy()

    history = get_prediction_history(full_engine)

    visible_history = history[
        history["cycle"] <= selected_cycle
    ].copy()

    current = visible_history.iloc[-1]

    rul = float(current["predicted_rul"])
    raw_rul = float(current["raw_rul"])

    st.subheader(f"Engine {engine_id} — Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Current Operating Cycle", selected_cycle)
    c2.metric("Estimated Remaining Life", f"{rul:.1f} cycles")
    c3.metric("Sensor Records", len(engine))

    if rul <= threshold:
        st.warning(
            f"Estimated RUL is below the "
            f"{threshold}-cycle warning threshold."
        )
    else:
        st.info(
            f"Estimated RUL is above the "
            f"{threshold}-cycle warning threshold."
        )

    with st.expander("Prediction Details"):
        st.write(f"Raw LSTM estimate: {raw_rul:.2f} cycles")
        st.write("Experimental buffer: 5 cycles")
        st.write(f"Adjusted estimate: {rul:.2f} cycles")

    st.divider()

    st.subheader("Remaining Useful Life History")

    rul_chart = (
        alt.Chart(visible_history)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "cycle:Q",
                title="Operating Cycle",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "predicted_rul:Q",
                title="Estimated RUL (cycles)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip(
                    "predicted_rul:Q",
                    title="Estimated RUL",
                    format=".2f",
                ),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(rul_chart, use_container_width=True)

    st.divider()

    st.subheader("Sensor Analysis")

    sensor = st.selectbox(
        "Select Sensor",
        checkpoint["sensor_names"],
        index=(
            checkpoint["sensor_names"].index("sensor_11")
            if "sensor_11" in checkpoint["sensor_names"]
            else 0
        ),
    )

    zoom = st.checkbox(
        "Zoom into sensor readings",
        value=True,
    )

    sensor_chart = (
        alt.Chart(engine)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "cycle:Q",
                title="Operating Cycle",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                f"{sensor}:Q",
                title="Sensor Reading",
                scale=alt.Scale(zero=not zoom),
            ),
            tooltip=[
                alt.Tooltip("cycle:Q"),
                alt.Tooltip(f"{sensor}:Q", format=".3f"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(sensor_chart, use_container_width=True)

    st.subheader("Recent Sensor Measurements")

    st.dataframe(
        engine[["cycle", sensor]].tail(10),
        hide_index=True,
        use_container_width=True,
    )


# ==========================================
# PAGE 2 — FLEET ANALYTICS
# ==========================================

elif page == "Fleet Analytics":

    st.title("Fleet Analytics")
    st.caption(
        "Fleet-wide remaining-life estimates and review prioritization"
    )

    fleet = get_fleet_predictions(df).copy()

    fleet["status"] = np.where(
        fleet["predicted_rul"] <= threshold,
        "Flagged for review",
        "Not flagged",
    )

    flagged = int(
        (fleet["status"] == "Flagged for review").sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Engines Analyzed", len(fleet))
    c2.metric("Flagged for Review", flagged)
    c3.metric(
        "Lowest Estimated RUL",
        f"{fleet['predicted_rul'].min():.1f} cycles",
    )

    st.divider()

    st.subheader("Engine Priority Ranking")

    st.caption(
        "Sorted by estimated remaining life. "
        "Lower estimates receive higher review priority."
    )

    table = fleet.copy()
    table["predicted_rul"] = table["predicted_rul"].round(2)

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("20 Engines with Lowest Estimated RUL")

    top20 = fleet.head(20).copy()

    chart = (
        alt.Chart(top20)
        .mark_bar()
        .encode(
            x=alt.X(
                "predicted_rul:Q",
                title="Estimated RUL (cycles)",
            ),
            y=alt.Y(
                "engine_id:N",
                title="Engine ID",
                sort=alt.EncodingSortField(
                    field="predicted_rul",
                    order="ascending",
                ),
            ),
            color=alt.Color(
                "status:N",
                title="Threshold Status",
                scale=alt.Scale(
                    domain=[
                        "Flagged for review",
                        "Not flagged",
                    ],
                    range=["#EF4444", "#3B82F6"],
                ),
            ),
            tooltip=[
                alt.Tooltip("engine_id:N", title="Engine"),
                alt.Tooltip(
                    "predicted_rul:Q",
                    format=".2f",
                ),
                alt.Tooltip("status:N"),
            ],
        )
        .properties(height=550)
    )

    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "Download Fleet Predictions",
        data=fleet.to_csv(index=False).encode("utf-8"),
        file_name="fleet_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Fleet priorities are experimental estimates, "
        "not verified maintenance recommendations."
    )


# ==========================================
# PAGE 3 — MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")
    st.caption(
        "NASA FD001 official test set — 100 unseen engines"
    )

    evaluation = load_evaluation()

    actual = evaluation["true_RUL"].to_numpy()
    near_mask = actual <= 30

    metrics = []

    for name in ["XGBoost", "LSTM", "LSTM + 5"]:

        predicted = evaluation[name].to_numpy()
        errors = predicted - actual

        metrics.append({
            "Model": name,
            "MAE": np.mean(np.abs(errors)),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "Overestimation %": np.mean(errors > 0) * 100,
            "Near Failure MAE": np.mean(
                np.abs(errors[near_mask])
            ),
            "Worst Absolute Error": np.max(np.abs(errors)),
        })

    metrics_df = pd.DataFrame(metrics)

    st.subheader("Performance Comparison")

    st.dataframe(
        metrics_df.round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Mean Absolute Error")

    mae_chart = (
        alt.Chart(metrics_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model:N",
                sort=["XGBoost", "LSTM", "LSTM + 5"],
            ),
            y=alt.Y(
                "MAE:Q",
                title="MAE (cycles)",
            ),
            color=alt.Color("Model:N", legend=None),
            tooltip=[
                "Model:N",
                alt.Tooltip("MAE:Q", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(mae_chart, use_container_width=True)

    st.divider()

    st.subheader("Actual vs Predicted RUL")

    selected_model = st.selectbox(
        "Compare Predictions",
        ["XGBoost", "LSTM", "LSTM + 5"],
        index=2,
    )

    limit = float(max(
        evaluation["true_RUL"].max(),
        evaluation[selected_model].max(),
    ) + 10)

    scatter = (
        alt.Chart(evaluation)
        .mark_circle(size=65, opacity=0.65)
        .encode(
            x=alt.X(
                "true_RUL:Q",
                title="Actual RUL",
                scale=alt.Scale(domain=[0, limit]),
            ),
            y=alt.Y(
                field=selected_model,
                type="quantitative",
                title="Predicted RUL",
                scale=alt.Scale(domain=[0, limit]),
            ),
            tooltip=[
                alt.Tooltip("engine_id:Q", title="Engine"),
                alt.Tooltip("true_RUL:Q", title="Actual RUL"),
                alt.Tooltip(
                    field=selected_model,
                    type="quantitative",
                    title="Predicted RUL",
                    format=".2f",
                ),
            ],
        )
    )

    reference = pd.DataFrame({
        "actual": [0, limit],
        "predicted": [0, limit],
    })

    perfect_line = (
        alt.Chart(reference)
        .mark_line(
            color="#EF4444",
            strokeDash=[6, 4],
        )
        .encode(
            x="actual:Q",
            y="predicted:Q",
        )
    )

    st.altair_chart(
        (scatter + perfect_line).properties(height=400),
        use_container_width=True,
    )

    st.caption(
        "The dashed line represents perfect prediction. "
        "Points above it overestimate remaining life."
    )

    st.divider()

    st.subheader("Model Limitations")

    final_errors = evaluation["LSTM + 5"].to_numpy() - actual

    worst_idx = int(np.argmax(np.abs(final_errors)))
    worst_engine = int(evaluation.iloc[worst_idx]["engine_id"])

    c1, c2 = st.columns(2)

    c1.metric(
        "Largest Absolute Error",
        f"{np.max(np.abs(final_errors)):.2f} cycles",
    )

    c2.metric(
        "Errors Exceeding 30 Cycles",
        f"{np.sum(np.abs(final_errors) > 30)} / 100",
    )

    st.write(f"Worst prediction occurred on Engine {worst_engine}.")

    st.warning(
        "The system can substantially overestimate remaining "
        "engine life. These predictions must not be used for "
        "real maintenance decisions."
    )

    st.info(
        "The 5-cycle buffer was selected using validation data. "
        "Because official test results were subsequently inspected "
        "during model selection, the reported test metrics are "
        "descriptive rather than a fully independent final "
        "model-selection assessment."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Predictive Maintenance AI | NASA C-MAPSS FD001 | "
    "PyTorch LSTM + XGBoost | Experimental research prototype"
)
