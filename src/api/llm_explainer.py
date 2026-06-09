from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical data assistant. Summarize mortality risk predictions "
            "in plain English for clinicians. Be concise, factual, and avoid definitive "
            "diagnoses. Mention the most influential vitals/labs and whether they "
            "increased or decreased estimated risk.",
        ),
        (
            "human",
            "Mortality risk score: {risk_score:.1%}\n"
            "Top contributing features (positive SHAP increases risk, negative decreases):\n"
            "{feature_summary}\n\n"
            "Write a 2-3 sentence plain-English explanation.",
        ),
    ]
)


def _format_feature_summary(shap_features: list[dict], top_n: int = 5) -> str:
    lines = []
    for item in shap_features[:top_n]:
        direction = "increased" if item["shap_value"] > 0 else "decreased"
        lines.append(
            f"- {item['feature']}: SHAP {item['shap_value']:+.4f} ({direction} risk)"
        )
    return "\n".join(lines)


def _fallback_explanation(risk_score: float, shap_features: list[dict]) -> str:
    top = shap_features[:3]
    drivers = []
    for item in top:
        direction = "raised" if item["shap_value"] > 0 else "lowered"
        readable = item["feature"].replace("_", " ")
        drivers.append(f"{readable} {direction} the estimated risk")

    driver_text = "; ".join(drivers)
    return (
        f"The model estimates a {risk_score:.1%} in-hospital mortality risk. "
        f"The strongest drivers were: {driver_text}. "
        "This explanation is template-based because no LLM API key was configured."
    )


def generate_explanation(risk_score: float, shap_features: list[dict]) -> str:
    if not settings.openai_api_key:
        return _fallback_explanation(risk_score, shap_features)

    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.2)
    chain = PROMPT | llm
    response = chain.invoke(
        {
            "risk_score": risk_score,
            "feature_summary": _format_feature_summary(shap_features),
        }
    )
    return response.content.strip()
