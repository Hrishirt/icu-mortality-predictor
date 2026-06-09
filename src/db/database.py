from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import settings

Base = declarative_base()


class PredictionLog(Base):
    # Audit trail for every API prediction.
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    patient_id = Column(String(64), nullable=True)
    model_name = Column(String(64), nullable=False)
    risk_score = Column(Float, nullable=False)
    input_features = Column(JSON, nullable=False)
    shap_features = Column(JSON, nullable=False)
    explanation = Column(String, nullable=False)


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def log_prediction(
    *,
    patient_id: str | None,
    model_name: str,
    risk_score: float,
    input_features: dict,
    shap_features: list[dict],
    explanation: str,
    session: Session | None = None,
) -> PredictionLog:
    owns_session = session is None
    session = session or get_session()
    record = PredictionLog(
        patient_id=patient_id,
        model_name=model_name,
        risk_score=risk_score,
        input_features=input_features,
        shap_features=shap_features,
        explanation=explanation,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    if owns_session:
        session.close()
    return record
