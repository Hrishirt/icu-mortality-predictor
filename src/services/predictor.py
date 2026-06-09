from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch  # noqa: F401

from src.config import settings
from src.explainability.shap_explainer import SHAPExplainer


class MortalityPredictor:
    def __init__(self, model_dir: Path | None = None, model_name: str | None = None):
        self.model_dir = model_dir or settings.model_dir
        self.model_name = model_name or settings.default_model
        self.feature_names = self._load_feature_names()
        self.model = self._load_model()
        self.shap_explainer = SHAPExplainer.from_artifacts(
            self.model_dir,
            model_type=self.model_name,
        )

    def _load_feature_names(self) -> list[str]:
        with open(self.model_dir / "feature_names.json", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self):
        if self.model_name == "gradient_boosting":
            return joblib.load(self.model_dir / "gradient_boosting.joblib")
        if self.model_name == "mlp":
            return joblib.load(self.model_dir / "mlp.joblib")
        raise ValueError(f"Unknown model: {self.model_name}")

    def _vectorize(self, features: dict[str, float]) -> np.ndarray:
        # Align input dict to the trained feature order; missing keys become NaN.
        values = [features.get(name, np.nan) for name in self.feature_names]
        return np.array(values, dtype=float).reshape(1, -1)

    def predict(self, features: dict[str, float]) -> tuple[float, list[dict]]:
        vector = self._vectorize(features)
        risk_score = float(self.model.predict_proba(vector)[0, 1])
        shap_features = self.shap_explainer.explain(vector)
        return risk_score, shap_features
