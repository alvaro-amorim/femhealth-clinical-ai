"""Prepared final holdout evaluation for the frozen selection."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from femhealth.data import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES
from femhealth.final_selection import SELECTED_THRESHOLD, SELECTED_VARIANT, build_selected_estimator
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL

DEVELOPMENT_SAMPLE_COUNT = 455
FINAL_TEST_SAMPLE_COUNT = 114
DEVELOPMENT_CLASS_DISTRIBUTION = {
    0: 170,
    1: 285,
}
FINAL_TEST_CLASS_DISTRIBUTION = {
    0: 42,
    1: 72,
}


def evaluate_final_holdout(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator=None,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Evaluate the frozen selection on final holdout data."""
    _validate_final_holdout_inputs(X_development, y_development, X_test, y_test)

    selected_estimator = build_selected_estimator() if estimator is None else estimator
    fitted_estimator = clone(selected_estimator)
    fitted_estimator.fit(X_development, y_development)

    probability_malignant = pd.Series(
        _predict_malignant_probabilities(fitted_estimator, X_test),
        index=X_test.index,
        name="probability_malignant",
    )
    predicted_label = _apply_selected_threshold(probability_malignant)
    predictions = _build_predictions_frame(y_test, probability_malignant, predicted_label)
    summary = _build_summary(y_test, probability_malignant, predicted_label)

    return summary, predictions, fitted_estimator


def _validate_final_holdout_inputs(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    _validate_dataset("development", X_development, y_development)
    _validate_dataset("test", X_test, y_test)

    if len(X_development) != DEVELOPMENT_SAMPLE_COUNT:
        raise ValueError("Unexpected development sample count")

    if len(X_test) != FINAL_TEST_SAMPLE_COUNT:
        raise ValueError("Unexpected final test sample count")

    if y_development.value_counts().sort_index().to_dict() != DEVELOPMENT_CLASS_DISTRIBUTION:
        raise ValueError("Unexpected development class distribution")

    if y_test.value_counts().sort_index().to_dict() != FINAL_TEST_CLASS_DISTRIBUTION:
        raise ValueError("Unexpected final test class distribution")

    if not set(X_development.index).isdisjoint(set(X_test.index)):
        raise ValueError("Development and final test indices must be disjoint")


def _validate_dataset(name: str, X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"{name} X must be a DataFrame")

    if not isinstance(y, pd.Series):
        raise ValueError(f"{name} y must be a Series")

    if len(X) != len(y):
        raise ValueError(f"{name} length mismatch")

    if X.shape[1] != WDBC_FEATURE_COUNT:
        raise ValueError(f"{name} unexpected feature count")

    if list(X.columns) != WDBC_FEATURE_NAMES:
        raise ValueError(f"{name} unexpected feature columns")

    if not X.index.equals(y.index):
        raise ValueError(f"{name} index mismatch")

    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"{name} null values found")

    if not all(is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError(f"{name} features must be numeric")

    if set(y.unique()) != {MALIGNANT_LABEL, BENIGN_LABEL}:
        raise ValueError(f"{name} unexpected target classes")


def _predict_malignant_probabilities(estimator: object, X: pd.DataFrame):
    probabilities = estimator.predict_proba(X)
    malignant_index = list(estimator.classes_).index(MALIGNANT_LABEL)
    return probabilities[:, malignant_index]


def _apply_selected_threshold(probability_malignant: pd.Series) -> pd.Series:
    return pd.Series(
        [
            MALIGNANT_LABEL if probability >= SELECTED_THRESHOLD else BENIGN_LABEL
            for probability in probability_malignant
        ],
        index=probability_malignant.index,
        name="predicted_label",
    )


def _build_predictions_frame(
    y_true: pd.Series,
    probability_malignant: pd.Series,
    predicted_label: pd.Series,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "true_label": y_true,
            "probability_malignant": probability_malignant,
            "predicted_label": predicted_label,
        },
        index=y_true.index,
    )
    predictions["correct"] = predictions["true_label"].eq(predictions["predicted_label"])
    predictions["error_type"] = [
        _classify_error_type(true_label, predicted)
        for true_label, predicted in zip(
            predictions["true_label"],
            predictions["predicted_label"],
            strict=True,
        )
    ]
    return predictions


def _classify_error_type(true_label: int, predicted_label: int) -> str:
    if true_label == predicted_label:
        return "correct"

    if true_label == MALIGNANT_LABEL and predicted_label == BENIGN_LABEL:
        return "false_negative_malignant"

    return "false_positive_malignant"


def _build_summary(
    y_true: pd.Series,
    probability_malignant: pd.Series,
    predicted_label: pd.Series,
) -> pd.DataFrame:
    malignant_target = y_true.eq(MALIGNANT_LABEL).astype(int)
    counts = _build_confusion_counts(y_true, predicted_label)

    return pd.DataFrame(
        [
            {
                "selected_variant": SELECTED_VARIANT,
                "threshold": SELECTED_THRESHOLD,
                "test_sample_count": len(y_true),
                "accuracy": accuracy_score(y_true, predicted_label),
                "balanced_accuracy": balanced_accuracy_score(y_true, predicted_label),
                "precision_malignant": precision_score(
                    y_true,
                    predicted_label,
                    pos_label=MALIGNANT_LABEL,
                    zero_division=0,
                ),
                "recall_malignant": recall_score(
                    y_true,
                    predicted_label,
                    pos_label=MALIGNANT_LABEL,
                ),
                "f1_malignant": f1_score(
                    y_true,
                    predicted_label,
                    pos_label=MALIGNANT_LABEL,
                ),
                "specificity_benign": recall_score(
                    y_true,
                    predicted_label,
                    pos_label=BENIGN_LABEL,
                ),
                "roc_auc_malignant": roc_auc_score(malignant_target, probability_malignant),
                "average_precision_malignant": average_precision_score(
                    malignant_target,
                    probability_malignant,
                ),
                "brier_score": brier_score_loss(malignant_target, probability_malignant),
                "log_loss": log_loss(malignant_target, probability_malignant, labels=[0, 1]),
                **counts,
            }
        ]
    )


def _build_confusion_counts(y_true: pd.Series, predicted_label: pd.Series) -> dict[str, int]:
    malignant = y_true.eq(MALIGNANT_LABEL)
    benign = y_true.eq(BENIGN_LABEL)
    predicted_malignant = predicted_label.eq(MALIGNANT_LABEL)
    predicted_benign = predicted_label.eq(BENIGN_LABEL)

    return {
        "true_malignant": int((malignant & predicted_malignant).sum()),
        "false_negative_malignant": int((malignant & predicted_benign).sum()),
        "false_positive_malignant": int((benign & predicted_malignant).sum()),
        "true_benign": int((benign & predicted_benign).sum()),
    }
