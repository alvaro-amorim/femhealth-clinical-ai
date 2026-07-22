"""Baseline cross-validation evaluation for candidate models."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from femhealth.data import WDBC_FEATURE_COUNT, WDBC_FEATURE_NAMES
from femhealth.model_pipelines import build_candidate_pipelines

CV_SPLITS = 5
RANDOM_STATE = 42
MALIGNANT_LABEL = 0
BENIGN_LABEL = 1

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "precision_malignant",
    "recall_malignant",
    "f1_malignant",
    "specificity_benign",
    "roc_auc_malignant",
    "average_precision_malignant",
]


def build_stratified_cv() -> StratifiedKFold:
    """Build a fresh stratified 5-fold cross-validator."""
    return StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def _get_malignant_scores(estimator, X: pd.DataFrame):
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        class_index = list(estimator.classes_).index(MALIGNANT_LABEL)
        return probabilities[:, class_index]

    decision_scores = estimator.decision_function(X)
    classes = list(estimator.classes_)

    if getattr(decision_scores, "ndim", 1) == 2:
        class_index = classes.index(MALIGNANT_LABEL)
        return decision_scores[:, class_index]

    if classes[1] == MALIGNANT_LABEL:
        return decision_scores

    return -decision_scores


def _roc_auc_malignant_scorer(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    malignant_target = y.eq(MALIGNANT_LABEL).astype(int)
    return roc_auc_score(malignant_target, _get_malignant_scores(estimator, X))


def _average_precision_malignant_scorer(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    malignant_target = y.eq(MALIGNANT_LABEL).astype(int)
    return average_precision_score(malignant_target, _get_malignant_scores(estimator, X))


def evaluate_baseline_candidates(
    X_development: pd.DataFrame,
    y_development: pd.Series,
) -> pd.DataFrame:
    """Evaluate candidate pipelines with stratified cross-validation."""
    validate_development_data(X_development, y_development)

    pipelines = build_candidate_pipelines()
    scorers = build_evaluation_scorers()
    rows = []

    for model_name, pipeline in pipelines.items():
        cv_scores = cross_validate(
            pipeline,
            X_development,
            y_development,
            cv=build_stratified_cv(),
            scoring=scorers,
            error_score="raise",
        )
        predictions = cross_val_predict(
            pipeline,
            X_development,
            y_development,
            cv=build_stratified_cv(),
        )

        row = {"model": model_name}
        for metric_name in METRIC_NAMES:
            metric_scores = pd.Series(cv_scores[f"test_{metric_name}"])
            row[f"{metric_name}_mean"] = metric_scores.mean()
            row[f"{metric_name}_std"] = metric_scores.std(ddof=0)

        predicted_series = pd.Series(predictions, index=y_development.index)
        row.update(_build_confusion_counts(y_development, predicted_series))
        rows.append(row)

    return pd.DataFrame(rows)


def validate_development_data(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame")

    if not isinstance(y, pd.Series):
        raise ValueError("y must be a Series")

    if len(X) != len(y):
        raise ValueError("Length mismatch")

    if X.shape[1] != WDBC_FEATURE_COUNT:
        raise ValueError("Unexpected feature count")

    if list(X.columns) != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected feature columns")

    if not X.index.equals(y.index):
        raise ValueError("Index mismatch")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Null values found")

    if not all(is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Features must be numeric")

    if set(y.unique()) != {MALIGNANT_LABEL, BENIGN_LABEL}:
        raise ValueError("Unexpected target classes")


def build_evaluation_scorers() -> dict[str, str | object]:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision_malignant": make_scorer(
            precision_score,
            pos_label=MALIGNANT_LABEL,
            zero_division=0,
        ),
        "recall_malignant": make_scorer(recall_score, pos_label=MALIGNANT_LABEL),
        "f1_malignant": make_scorer(f1_score, pos_label=MALIGNANT_LABEL),
        "specificity_benign": make_scorer(recall_score, pos_label=BENIGN_LABEL),
        "roc_auc_malignant": _roc_auc_malignant_scorer,
        "average_precision_malignant": _average_precision_malignant_scorer,
    }


def _build_confusion_counts(y_true: pd.Series, y_predicted: pd.Series) -> dict[str, int]:
    malignant = y_true.eq(MALIGNANT_LABEL)
    benign = y_true.eq(BENIGN_LABEL)
    predicted_malignant = y_predicted.eq(MALIGNANT_LABEL)
    predicted_benign = y_predicted.eq(BENIGN_LABEL)

    return {
        "true_malignant": int((malignant & predicted_malignant).sum()),
        "false_negative_malignant": int((malignant & predicted_benign).sum()),
        "false_positive_malignant": int((benign & predicted_malignant).sum()),
        "true_benign": int((benign & predicted_benign).sum()),
    }
