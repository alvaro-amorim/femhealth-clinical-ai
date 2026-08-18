"""Pydantic schemas for the FemHealth inference API."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from femhealth.data import WDBC_FEATURE_NAMES

ACADEMIC_DISCLAIMER = (
    "Resultado de classificação acadêmica. Não é diagnóstico médico e não "
    "substitui avaliação por profissional de saúde."
)

PREDICTION_FEATURE_EXAMPLE = {
    "mean radius": 14.0,
    "mean texture": 19.0,
    "mean perimeter": 92.0,
    "mean area": 650.0,
    "mean smoothness": 0.10,
    "mean compactness": 0.11,
    "mean concavity": 0.09,
    "mean concave points": 0.05,
    "mean symmetry": 0.18,
    "mean fractal dimension": 0.06,
    "radius error": 0.50,
    "texture error": 1.00,
    "perimeter error": 3.00,
    "area error": 40.0,
    "smoothness error": 0.007,
    "compactness error": 0.020,
    "concavity error": 0.030,
    "concave points error": 0.010,
    "symmetry error": 0.020,
    "fractal dimension error": 0.003,
    "worst radius": 16.0,
    "worst texture": 25.0,
    "worst perimeter": 105.0,
    "worst area": 850.0,
    "worst smoothness": 0.14,
    "worst compactness": 0.25,
    "worst concavity": 0.28,
    "worst concave points": 0.12,
    "worst symmetry": 0.30,
    "worst fractal dimension": 0.08,
}

PREDICTION_REQUEST_EXAMPLE = {"features": PREDICTION_FEATURE_EXAMPLE}
PREDICTION_RESPONSE_EXAMPLE = {
    "probability_malignant": 0.31,
    "probability_benign": 0.69,
    "predicted_label": 1,
    "predicted_class": "benign",
    "threshold": 0.51,
    "disclaimer": ACADEMIC_DISCLAIMER,
}


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    artifact_version: str
    selected_variant: str


class ModelInfoResponse(BaseModel):
    artifact_version: str
    selected_variant: str
    selected_model: str
    selected_calibration: str
    threshold: float
    class_labels: dict[str, int]
    feature_count: int
    feature_names: list[str]
    training_sample_count: int
    model_sha256: str
    final_holdout_metrics: dict[str, str | int | float]
    disclaimer: str


class ExplainabilityFeatureResponse(BaseModel):
    rank: int
    feature_name: str
    feature_position: int
    mean_importance: float
    std_importance: float
    median_importance: float
    min_importance: float
    max_importance: float
    positive_fraction: float
    fold_count: int
    observation_count: int


class ExplainabilityFoldScoreResponse(BaseModel):
    fold: int
    train_sample_count: int
    validation_sample_count: int
    train_malignant_count: int
    train_benign_count: int
    validation_malignant_count: int
    validation_benign_count: int
    baseline_roc_auc: float


class ExplainabilityResponse(BaseModel):
    method: str
    scorer: str
    selected_variant: str
    selected_model: str
    selected_calibration: str
    selected_threshold: float
    development_sample_count: int
    cv_splits: int
    permutation_repeats: int
    feature_count: int
    detail_row_count: int
    holdout_used: bool
    mean_fold_roc_auc: float
    std_fold_roc_auc: float
    features: list[ExplainabilityFeatureResponse]
    fold_scores: list[ExplainabilityFoldScoreResponse]
    limitations: list[str]
    plot_endpoint: str
    disclaimer: str


class DemoCaseResponse(BaseModel):
    case_id: str
    sample_index: int
    reference_label: Literal[0, 1]
    reference_class: Literal["malignant", "benign"]
    features: dict[str, float]


class DemoCasesResponse(BaseModel):
    artifact_version: str
    source_dataset: str
    source_split: str
    selection_rule: str
    used_for_training: bool
    used_for_model_selection: bool
    created_after_final_evaluation: bool
    training_sample_count: int
    holdout_sample_count: int
    official_holdout_accuracy: float
    case_count: int
    malignant_case_count: int
    benign_case_count: int
    feature_names: list[str]
    cases: list[DemoCaseResponse]
    disclaimer: str


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": PREDICTION_REQUEST_EXAMPLE},
    )

    features: dict[str, float] = Field(
        description=(
            "Mapa com exatamente as 30 features canônicas do WDBC. Os nomes, valores "
            "finitos e a presença de todas as chaves são validados antes da inferência."
        ),
        examples=[PREDICTION_FEATURE_EXAMPLE],
    )

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("features must be an object")

        expected_features = set(WDBC_FEATURE_NAMES)
        received_features = set(value)
        missing_features = [feature for feature in WDBC_FEATURE_NAMES if feature not in value]
        extra_features = sorted(received_features - expected_features)

        if missing_features:
            raise ValueError("Missing required features")

        if extra_features:
            raise ValueError("Unexpected extra features")

        ordered_features = {}
        for feature_name in WDBC_FEATURE_NAMES:
            raw_value = value[feature_name]
            if (
                raw_value is None
                or isinstance(raw_value, bool | str)
                or not isinstance(raw_value, int | float)
            ):
                raise ValueError("Feature values must be finite numbers")

            numeric_value = float(raw_value)
            if not math.isfinite(numeric_value):
                raise ValueError("Feature values must be finite numbers")

            ordered_features[feature_name] = numeric_value

        return ordered_features


class PredictionResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": PREDICTION_RESPONSE_EXAMPLE})

    probability_malignant: float = Field(
        description="Probabilidade produzida pelo classificador para a classe malignant (0)."
    )
    probability_benign: float = Field(
        description="Probabilidade produzida pelo classificador para a classe benign (1)."
    )
    predicted_label: int = Field(
        description="Rótulo acadêmico previsto: 0 para malignant e 1 para benign."
    )
    predicted_class: Literal["malignant", "benign"] = Field(
        description="Classe acadêmica prevista a partir do threshold congelado."
    )
    threshold: float = Field(
        description="Threshold maligno congelado aplicado pelo classificador (0.51)."
    )
    disclaimer: str = Field(
        description="Aviso de uso responsável: o projeto não possui validade clínica."
    )
