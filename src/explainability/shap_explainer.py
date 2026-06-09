from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def _ensure_native_runtime() -> None:
    import torch  # noqa: F401  # required import order on Windows


def _import_shap():
    _ensure_native_runtime()
    import shap

    return shap


class SHAPExplainer:
    def __init__(
        self,
        model,
        feature_names: list[str],
        background_data: np.ndarray,
        model_type: str = "gradient_boosting",
    ):
        shap = _import_shap()
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type

        if model_type == "gradient_boosting":
            # TreeExplainer is fast and exact for tree models.
            classifier = model.named_steps["classifier"]
            imputer = model.named_steps["imputer"]
            background_imputed = imputer.transform(background_data)
            self.explainer = shap.TreeExplainer(classifier)
            self._transform = imputer.transform
            self._background = background_imputed
        else:
            self.explainer = shap.KernelExplainer(
                self._mlp_predict_proba,
                background_data[:50],
            )
            self._transform = None

    def _mlp_predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def explain(self, patient_features: np.ndarray) -> list[dict[str, float]]:
        if patient_features.ndim == 1:
            patient_features = patient_features.reshape(1, -1)

        if self.model_type == "gradient_boosting":
            X_imputed = self._transform(patient_features)
            shap_values = self.explainer.shap_values(X_imputed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            values = shap_values[0]
        else:
            shap_values = self.explainer.shap_values(patient_features, nsamples=100)
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        contributions = [
            {"feature": name, "shap_value": float(value)}
            for name, value in zip(self.feature_names, values, strict=True)
        ]
        # Largest absolute contributions are the most influential features.
        contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        return contributions

    @classmethod
    def from_artifacts(
        cls,
        model_dir: Path,
        model_type: str = "gradient_boosting",
    ) -> "SHAPExplainer":
        import json

        _ensure_native_runtime()

        with open(model_dir / "feature_names.json", encoding="utf-8") as f:
            feature_names = json.load(f)

        background = np.load(model_dir / "shap_background.npy")

        if model_type == "gradient_boosting":
            model = joblib.load(model_dir / "gradient_boosting.joblib")
        else:
            model = joblib.load(model_dir / "mlp.joblib")

        return cls(model, feature_names, background, model_type=model_type)
