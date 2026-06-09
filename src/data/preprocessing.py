from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Demographics recorded once at ICU admission.
STATIC_PARAMS = {"Age", "Gender", "Height", "Weight", "ICUType", "RecordID"}
# Summary stats computed for each vital/lab time series.
AGG_FUNCS = ("mean", "median", "min", "max", "std", "last", "count")


def load_outcomes(outcomes_path: Path) -> pd.DataFrame:
    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes.rename(columns={"In-hospital_death": "mortality"})
    return outcomes


def load_patient_record(record_path: Path) -> pd.DataFrame:
    df = pd.read_csv(record_path)
    df["Time"] = pd.to_timedelta("0:" + df["Time"].astype(str))
    return df


def _aggregate_timeseries(series: pd.Series) -> dict[str, float]:
    # -1 is a missing-value sentinel in the PhysioNet files.
    values = pd.to_numeric(series, errors="coerce").replace(-1, np.nan).dropna()
    if values.empty:
        return {func: np.nan for func in AGG_FUNCS}

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
        "last": float(values.iloc[-1]),
        "count": float(len(values)),
    }


def extract_patient_features(record_df: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}
    admission = record_df[record_df["Time"] == pd.Timedelta(0)]

    # Static patient attributes from the admission row.
    for param in STATIC_PARAMS - {"RecordID"}:
        row = admission[admission["Parameter"] == param]
        value = row["Value"].iloc[0] if not row.empty else np.nan
        features[param] = float(value) if pd.notna(value) else np.nan

    # Vitals and labs measured over the ICU stay.
    timeseries = record_df[~record_df["Parameter"].isin(STATIC_PARAMS)]
    for param, group in timeseries.groupby("Parameter"):
        for func, value in _aggregate_timeseries(group["Value"]).items():
            features[f"{param}_{func}"] = value

    return features


def build_feature_matrix(data_dir: Path, outcomes_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    set_a_dir = data_dir / "set-a"
    if outcomes_path is None:
        outcomes_path = data_dir / "Outcomes-a.txt"

    outcomes = load_outcomes(outcomes_path)
    records = sorted(set_a_dir.glob("*.txt"))

    rows: list[dict[str, float]] = []
    record_ids: list[int] = []

    for record_path in records:
        record_id = int(record_path.stem)
        record_df = load_patient_record(record_path)
        features = extract_patient_features(record_df)
        features["RecordID"] = record_id
        rows.append(features)
        record_ids.append(record_id)

    feature_df = pd.DataFrame(rows).set_index("RecordID")
    labels = outcomes.set_index("RecordID").loc[record_ids, "mortality"]
    feature_df = feature_df.replace(-1, np.nan)
    return feature_df, labels
