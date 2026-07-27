"""Global explainability by cross-validated permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.final_evaluation import DEVELOPMENT_CLASS_DISTRIBUTION, DEVELOPMENT_SAMPLE_COUNT
from femhealth.final_selection import build_selected_estimator
from femhealth.model_evaluation import (
    BENIGN_LABEL,
    MALIGNANT_LABEL,
    build_stratified_cv,
    validate_development_data,
)

EXPLAINABILITY_RANDOM_STATE = 42
EXPLAINABILITY_CV_SPLITS = 5
PERMUTATION_REPEATS = 10
EXPLAINABILITY_SCORER = "roc_auc_malignant"


def get_malignant_probabilities(
    estimator,
    X: pd.DataFrame,
) -> np.ndarray:
    """Extract P(malignant=0) from a fitted estimator."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Estimator must support predict_proba")

    classes = list(getattr(estimator, "classes_", []))
    if MALIGNANT_LABEL not in classes:
        raise ValueError("Estimator does not contain malignant class")

    probabilities = np.asarray(estimator.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(X):
        raise ValueError("Unexpected probability shape")

    if probabilities.shape[1] != len(classes):
        raise ValueError("Unexpected probability class count")

    malignant_index = classes.index(MALIGNANT_LABEL)
    return probabilities[:, malignant_index]


def score_malignant_roc_auc(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Score ROC AUC using malignant probability as the positive score."""
    malignant_target = y.eq(MALIGNANT_LABEL).astype(int)
    return roc_auc_score(malignant_target, get_malignant_probabilities(estimator, X))


def compute_cross_validated_permutation_importance(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    estimator_factory=None,
    n_repeats: int = PERMUTATION_REPEATS,
    random_state: int = EXPLAINABILITY_RANDOM_STATE,
    n_jobs: int | None = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute permutation importance only on validation folds."""
    _validate_explainability_inputs(X_development, y_development, n_repeats)
    selected_estimator_factory = (
        build_selected_estimator if estimator_factory is None else estimator_factory
    )
    cv = build_stratified_cv()
    if cv.get_n_splits() != EXPLAINABILITY_CV_SPLITS:
        raise RuntimeError("Unexpected explainability CV split count")

    detail_rows = []
    fold_rows = []

    for fold, (train_positions, validation_positions) in enumerate(
        cv.split(X_development, y_development),
        start=1,
    ):
        X_train = X_development.iloc[train_positions]
        y_train = y_development.iloc[train_positions]
        X_validation = X_development.iloc[validation_positions]
        y_validation = y_development.iloc[validation_positions]

        estimator = selected_estimator_factory()
        estimator.fit(X_train, y_train)

        baseline_roc_auc = score_malignant_roc_auc(estimator, X_validation, y_validation)
        validation_counts = y_validation.value_counts().sort_index().to_dict()
        train_counts = y_train.value_counts().sort_index().to_dict()

        fold_rows.append(
            {
                "fold": fold,
                "train_sample_count": len(X_train),
                "validation_sample_count": len(X_validation),
                "train_malignant_count": train_counts[MALIGNANT_LABEL],
                "train_benign_count": train_counts[BENIGN_LABEL],
                "validation_malignant_count": validation_counts[MALIGNANT_LABEL],
                "validation_benign_count": validation_counts[BENIGN_LABEL],
                "baseline_roc_auc": baseline_roc_auc,
            }
        )

        result = permutation_importance(
            estimator,
            X_validation,
            y_validation,
            scoring=score_malignant_roc_auc,
            n_repeats=n_repeats,
            random_state=random_state + fold,
            n_jobs=n_jobs,
        )

        for feature_index, feature_name in enumerate(WDBC_FEATURE_NAMES):
            for repeat_index, importance in enumerate(result.importances[feature_index], start=1):
                detail_rows.append(
                    {
                        "fold": fold,
                        "feature_name": feature_name,
                        "feature_position": feature_index + 1,
                        "repeat": repeat_index,
                        "importance": importance,
                        "baseline_roc_auc": baseline_roc_auc,
                        "validation_sample_count": len(X_validation),
                        "validation_malignant_count": validation_counts[MALIGNANT_LABEL],
                        "validation_benign_count": validation_counts[BENIGN_LABEL],
                    }
                )

    details = pd.DataFrame(detail_rows)
    summary = _build_summary(details)
    fold_scores = pd.DataFrame(fold_rows)

    return details, summary, fold_scores


def _validate_explainability_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int,
) -> None:
    validate_development_data(X, y)

    if len(X) != DEVELOPMENT_SAMPLE_COUNT:
        raise ValueError("Unexpected development sample count")

    if y.value_counts().sort_index().to_dict() != DEVELOPMENT_CLASS_DISTRIBUTION:
        raise ValueError("Unexpected development class distribution")

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")


def _build_summary(details: pd.DataFrame) -> pd.DataFrame:
    summary = (
        details.groupby("feature_name", sort=False)
        .agg(
            feature_position=("feature_position", "first"),
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            median_importance=("importance", "median"),
            min_importance=("importance", "min"),
            max_importance=("importance", "max"),
            positive_fraction=("importance", lambda values: values.gt(0).mean()),
            fold_count=("fold", "nunique"),
            observation_count=("importance", "size"),
        )
        .reset_index()
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary
