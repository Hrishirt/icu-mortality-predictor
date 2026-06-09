from contextlib import asynccontextmanager

import torch  # noqa: F401 - initialize native runtime before SHAP/sklearn
from fastapi import FastAPI, HTTPException

from src.api.llm_explainer import generate_explanation
from src.api.schemas import PatientFeatures, PredictionResponse, SHAPContribution
from src.config import settings
from src.db.database import init_db, log_prediction
from src.services.predictor import MortalityPredictor

predictor: MortalityPredictor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor
    try:
        init_db()
    except Exception:
        pass
    if (settings.model_dir / "gradient_boosting.joblib").exists():
        predictor = MortalityPredictor()
    else:
        predictor = None
    yield


app = FastAPI(
    title="ICU Mortality Prediction API",
    description="PhysioNet 2012 in-hospital mortality risk with SHAP and LLM explanations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_name": settings.default_model,
    }


@app.get("/metrics")
def metrics():
    metrics_path = settings.model_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train models first.")
    import json

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/features")
def feature_names():
    features_path = settings.model_dir / "feature_names.json"
    if not features_path.exists():
        raise HTTPException(status_code=404, detail="Feature list not found. Train models first.")
    import json

    with open(features_path, encoding="utf-8") as f:
        features = json.load(f)
    return {"feature_count": len(features), "features": features}


@app.post("/predict", response_model=PredictionResponse)
def predict_mortality(payload: PatientFeatures):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run scripts/train_models.py first.",
        )

    risk_score, shap_features = predictor.predict(payload.features)
    explanation = generate_explanation(risk_score, shap_features)

    prediction_id = None
    try:
        record = log_prediction(
            patient_id=payload.patient_id,
            model_name=settings.default_model,
            risk_score=risk_score,
            input_features=payload.features,
            shap_features=shap_features,
            explanation=explanation,
        )
        prediction_id = record.id
    except Exception:
        pass

    return PredictionResponse(
        patient_id=payload.patient_id,
        model_name=settings.default_model,
        mortality_risk_score=risk_score,
        mortality_risk_percent=round(risk_score * 100, 2),
        top_shap_features=[SHAPContribution(**item) for item in shap_features[:10]],
        explanation=explanation,
        prediction_id=prediction_id,
    )
