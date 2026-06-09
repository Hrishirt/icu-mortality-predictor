from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data.preprocessing import build_feature_matrix
from src.models.gradient_boosting import train_gradient_boosting
from src.models.mlp import MortalityMLP, train_mlp


def evaluate_model(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    # Binary metrics at the default 0.5 decision threshold.
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _save_artifacts(
    output_dir: Path,
    *,
    feature_names: list[str],
    gb_model,
    mlp_model,
    metrics: dict,
    X_background: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gb_model, output_dir / "gradient_boosting.joblib")
    joblib.dump(mlp_model, output_dir / "mlp.joblib")

    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Background samples used by SHAP at inference time.
    np.save(output_dir / "shap_background.npy", X_background)

    mlp_predictor = mlp_model
    torch.save(
        {
            "state_dict": mlp_predictor.model.state_dict(),
            "input_dim": mlp_predictor.model.network[0].in_features,
            "hidden_dim": mlp_predictor.model.network[0].out_features,
            "imputer": mlp_predictor.imputer,
            "scaler": mlp_predictor.scaler,
        },
        output_dir / "mlp_torch.pt",
    )


def train_and_compare(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    data_dir = data_dir or settings.data_dir
    output_dir = output_dir or settings.model_dir

    X, y = build_feature_matrix(data_dir)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values,
    )

    gb_model = train_gradient_boosting(X_train, y_train, random_state=random_state)
    mlp_model = train_mlp(X_train, y_train, random_state=random_state)

    gb_probs = gb_model.predict_proba(X_test)[:, 1]
    mlp_probs = mlp_model.predict_proba(X_test)[:, 1]

    metrics = {
        "gradient_boosting": evaluate_model(y_test, gb_probs),
        "mlp": evaluate_model(y_test, mlp_probs),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "feature_count": len(feature_names),
    }

    _save_artifacts(
        output_dir,
        feature_names=feature_names,
        gb_model=gb_model,
        mlp_model=mlp_model,
        metrics=metrics,
        X_background=X_train[:200],
    )

    return metrics
