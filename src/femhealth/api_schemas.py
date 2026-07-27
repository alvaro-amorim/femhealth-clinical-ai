"""Pydantic schemas for the FemHealth inference API."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from femhealth.data import WDBC_FEATURE_NAMES

ACADEMIC_DISCLAIMER = (
    "Resultado de classificação acadêmica. Não é diagnóstico médico e não "
    "substitui avaliação por profissional de saúde."
)


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


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float]

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
    probability_malignant: float
    probability_benign: float
    predicted_label: int
    predicted_class: Literal["malignant", "benign"]
    threshold: float
    disclaimer: str
