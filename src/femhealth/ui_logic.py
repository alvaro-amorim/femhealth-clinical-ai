"""Pure presentation logic for the Streamlit interface."""

from __future__ import annotations

import math

import pandas as pd

from femhealth.ui_labels import FEATURE_LABELS_PT_BR


def validate_api_feature_contract(feature_names: list[str]) -> None:
    """Validate that API feature metadata matches the UI translation contract."""
    expected_features = list(FEATURE_LABELS_PT_BR)

    if len(feature_names) != len(expected_features):
        raise ValueError("Unexpected feature count")

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Duplicated feature names")

    if set(feature_names) != set(expected_features):
        raise ValueError("Unexpected feature names")

    if feature_names != expected_features:
        raise ValueError("Unexpected feature order")


def format_probability(value: float) -> str:
    """Format a probability as a Brazilian Portuguese percentage."""
    if not 0 <= value <= 1:
        raise ValueError("Probability must be between 0 and 1")

    return f"{value * 100:.2f}%".replace(".", ",")


def format_decimal_pt_br(
    value: float,
    decimal_places: int = 2,
) -> str:
    """Format a finite decimal number using Brazilian Portuguese separators."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Value must be a finite number")

    if not math.isfinite(float(value)):
        raise ValueError("Value must be a finite number")

    if decimal_places < 0:
        raise ValueError("Decimal places must be greater than or equal to zero")

    return f"{value:.{decimal_places}f}".replace(".", ",")


def model_variant_pt_br(selected_variant: str) -> str:
    """Translate selected model variant identifiers for presentation."""
    if selected_variant == "svm_sigmoid":
        return "SVM calibrado por sigmoid"

    raise ValueError("Unexpected selected variant")


def prediction_class_pt_br(predicted_class: str) -> str:
    """Translate API predicted classes for presentation."""
    if predicted_class == "malignant":
        return "Padrão classificado como maligno"

    if predicted_class == "benign":
        return "Padrão classificado como benigno"

    raise ValueError("Unexpected predicted class")


def build_confusion_matrix(final_metrics: dict) -> pd.DataFrame:
    """Build a display table from persisted confusion counts."""
    return pd.DataFrame(
        {
            "Previsto: maligno": [
                final_metrics["true_malignant"],
                final_metrics["false_positive_malignant"],
            ],
            "Previsto: benigno": [
                final_metrics["false_negative_malignant"],
                final_metrics["true_benign"],
            ],
        },
        index=["Real: maligno", "Real: benigno"],
    )
