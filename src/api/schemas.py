from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    patient_id: str | None = Field(default=None, description="Optional patient identifier")
    features: dict[str, float] = Field(
        ...,
        description="Feature name to value mapping matching trained model features",
    )


class SHAPContribution(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    patient_id: str | None
    model_name: str
    mortality_risk_score: float
    mortality_risk_percent: float
    top_shap_features: list[SHAPContribution]
    explanation: str
    prediction_id: int | None = None
