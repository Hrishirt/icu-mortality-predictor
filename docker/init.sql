CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    patient_id VARCHAR(64),
    model_name VARCHAR(64) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    input_features JSONB NOT NULL,
    shap_features JSONB NOT NULL,
    explanation TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs (created_at DESC);
