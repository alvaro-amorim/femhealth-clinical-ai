"""Pure presentation logic for the Streamlit interface."""

from __future__ import annotations

import math

import pandas as pd

from femhealth.ui_labels import FEATURE_LABELS_PT_BR, translate_feature_name


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


def build_explainability_feature_table(features: list[dict]) -> pd.DataFrame:
    """Build a display table for global permutation importance features."""
    if not features:
        raise ValueError("Explainability features must not be empty")

    ranks = [feature.get("rank") for feature in features]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicated explainability ranks")

    rows = []
    required_keys = {
        "rank",
        "feature_name",
        "mean_importance",
        "std_importance",
        "positive_fraction",
    }
    for feature in features:
        if not required_keys.issubset(feature):
            raise ValueError("Missing explainability feature values")

        feature_name = feature["feature_name"]
        rows.append(
            {
                "Posição no ranking": feature["rank"],
                "Variável": translate_feature_name(feature_name),
                "Chave canônica": feature_name,
                "Importância média": feature["mean_importance"],
                "Desvio-padrão": feature["std_importance"],
                "Fração positiva": feature["positive_fraction"],
            }
        )

    return pd.DataFrame(rows).sort_values("Posição no ranking").reset_index(drop=True)


def build_explainability_fold_table(fold_scores: list[dict]) -> pd.DataFrame:
    """Build a display table for explainability validation fold scores."""
    if not fold_scores:
        raise ValueError("Explainability fold scores must not be empty")

    required_keys = {
        "fold",
        "train_sample_count",
        "validation_sample_count",
        "validation_malignant_count",
        "validation_benign_count",
        "baseline_roc_auc",
    }
    rows = []
    for fold_score in fold_scores:
        if not required_keys.issubset(fold_score):
            raise ValueError("Missing explainability fold values")

        rows.append(
            {
                "Fold": fold_score["fold"],
                "Amostras de treinamento": fold_score["train_sample_count"],
                "Amostras de validação": fold_score["validation_sample_count"],
                "Malignos na validação": fold_score["validation_malignant_count"],
                "Benignos na validação": fold_score["validation_benign_count"],
                "ROC AUC maligno": fold_score["baseline_roc_auc"],
            }
        )

    return pd.DataFrame(rows).sort_values("Fold").reset_index(drop=True)
