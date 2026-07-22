"""Out-of-fold probability and threshold analysis on development data."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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

from femhealth.model_evaluation import (
    BENIGN_LABEL,
    MALIGNANT_LABEL,
    build_stratified_cv,
    validate_development_data,
)
from femhealth.model_tuning import build_tuned_candidate_pipelines

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_BINS = 10
MIN_RECALL_MALIGNANT = 0.97
DEFAULT_THRESHOLDS = tuple(value / 100 for value in range(5, 96))
PROBABILITY_VARIANTS = (
    "logistic_regression_native",
    "logistic_regression_sigmoid",
    "random_forest_native",
    "random_forest_sigmoid",
    "svm_sigmoid",
)


def build_probability_estimators() -> dict[str, object]:
    """Build fresh estimators for native and sigmoid-calibrated probabilities."""
    native_pipelines = build_tuned_candidate_pipelines()
    calibrated_pipelines = build_tuned_candidate_pipelines()

    return {
        "logistic_regression_native": native_pipelines["logistic_regression"],
        "logistic_regression_sigmoid": _build_sigmoid_calibrator(
            calibrated_pipelines["logistic_regression"]
        ),
        "random_forest_native": native_pipelines["random_forest"],
        "random_forest_sigmoid": _build_sigmoid_calibrator(calibrated_pipelines["random_forest"]),
        "svm_sigmoid": _build_sigmoid_calibrator(calibrated_pipelines["svm"]),
    }


def generate_oof_malignant_probabilities(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    estimators: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Generate out-of-fold probabilities for the malignant class."""
    validate_development_data(X_development, y_development)
    selected_estimators = build_probability_estimators() if estimators is None else estimators
    probabilities = pd.DataFrame(index=y_development.index)

    for variant, estimator in selected_estimators.items():
        variant_probabilities = pd.Series(index=y_development.index, dtype=float)
        prediction_counts = pd.Series(0, index=y_development.index, dtype=int)

        for train_positions, validation_positions in build_stratified_cv().split(
            X_development,
            y_development,
        ):
            fold_estimator = clone(estimator)
            X_train = X_development.iloc[train_positions]
            y_train = y_development.iloc[train_positions]
            X_validation = X_development.iloc[validation_positions]
            validation_index = y_development.iloc[validation_positions].index

            fold_estimator.fit(X_train, y_train)
            malignant_probabilities = _predict_malignant_probabilities(
                fold_estimator,
                X_validation,
            )

            variant_probabilities.loc[validation_index] = malignant_probabilities
            prediction_counts.loc[validation_index] += 1

        if not prediction_counts.eq(1).all():
            raise ValueError("Each record must receive one prediction")

        if not variant_probabilities.between(0, 1).all():
            raise ValueError("Probabilities must be between 0 and 1")

        probabilities[variant] = variant_probabilities

    return probabilities


def build_probability_quality_summary(
    y_development: pd.Series,
    oof_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize probability quality metrics by variant."""
    _validate_probability_inputs(y_development, oof_probabilities)
    malignant_target = y_development.eq(MALIGNANT_LABEL).astype(int)
    rows = []

    for variant in oof_probabilities.columns:
        variant_probabilities = oof_probabilities[variant]
        rows.append(
            {
                "variant": variant,
                "brier_score": brier_score_loss(malignant_target, variant_probabilities),
                "log_loss": log_loss(
                    malignant_target,
                    variant_probabilities,
                    labels=[0, 1],
                ),
                "roc_auc_malignant": roc_auc_score(malignant_target, variant_probabilities),
                "average_precision_malignant": average_precision_score(
                    malignant_target,
                    variant_probabilities,
                ),
            }
        )

    return pd.DataFrame(rows)


def build_calibration_table(
    y_development: pd.Series,
    oof_probabilities: pd.DataFrame,
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    """Build a quantile calibration table for malignant probabilities."""
    _validate_probability_inputs(y_development, oof_probabilities)
    malignant_target = y_development.eq(MALIGNANT_LABEL).astype(int)
    rows = []

    for variant in oof_probabilities.columns:
        observed_fractions, mean_probabilities = calibration_curve(
            malignant_target,
            oof_probabilities[variant],
            pos_label=1,
            n_bins=n_bins,
            strategy="quantile",
        )

        for bin_index, (mean_probability, observed_fraction) in enumerate(
            zip(mean_probabilities, observed_fractions, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "variant": variant,
                    "bin": bin_index,
                    "mean_predicted_probability": mean_probability,
                    "observed_malignant_fraction": observed_fraction,
                }
            )

    return pd.DataFrame(rows)


def build_threshold_table(
    y_development: pd.Series,
    oof_probabilities: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate operational thresholds for malignant probabilities."""
    _validate_probability_inputs(y_development, oof_probabilities)
    rows = []

    for variant in oof_probabilities.columns:
        for threshold in thresholds:
            predicted_labels = pd.Series(
                [
                    MALIGNANT_LABEL if probability >= threshold else BENIGN_LABEL
                    for probability in oof_probabilities[variant]
                ],
                index=y_development.index,
            )
            counts = _build_confusion_counts(y_development, predicted_labels)

            rows.append(
                {
                    "variant": variant,
                    "threshold": threshold,
                    "accuracy": accuracy_score(y_development, predicted_labels),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_development,
                        predicted_labels,
                    ),
                    "precision_malignant": precision_score(
                        y_development,
                        predicted_labels,
                        pos_label=MALIGNANT_LABEL,
                        zero_division=0,
                    ),
                    "recall_malignant": recall_score(
                        y_development,
                        predicted_labels,
                        pos_label=MALIGNANT_LABEL,
                    ),
                    "f1_malignant": f1_score(
                        y_development,
                        predicted_labels,
                        pos_label=MALIGNANT_LABEL,
                    ),
                    "specificity_benign": recall_score(
                        y_development,
                        predicted_labels,
                        pos_label=BENIGN_LABEL,
                    ),
                    **counts,
                }
            )

    return pd.DataFrame(rows)


def select_operating_points(
    threshold_table: pd.DataFrame,
    min_recall_malignant: float = MIN_RECALL_MALIGNANT,
) -> pd.DataFrame:
    """Select provisional academic operating points by variant."""
    rows = []

    for variant in PROBABILITY_VARIANTS:
        eligible = threshold_table[
            (threshold_table["variant"] == variant)
            & (threshold_table["recall_malignant"] >= min_recall_malignant)
        ]

        if eligible.empty:
            raise ValueError(f"No threshold reaches recall minimum for {variant}")

        selected = eligible.sort_values(
            by=[
                "specificity_benign",
                "f1_malignant",
                "precision_malignant",
                "threshold",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        rows.append(selected)

    return pd.DataFrame(rows).reset_index(drop=True)


def run_probability_analysis(
    X_development: pd.DataFrame,
    y_development: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Run out-of-fold probability, calibration, and threshold analysis."""
    oof_probabilities = generate_oof_malignant_probabilities(X_development, y_development)
    probability_quality = build_probability_quality_summary(y_development, oof_probabilities)
    calibration = build_calibration_table(y_development, oof_probabilities)
    thresholds = build_threshold_table(y_development, oof_probabilities)
    operating_points = select_operating_points(thresholds)

    return {
        "oof_probabilities": oof_probabilities,
        "probability_quality": probability_quality,
        "calibration": calibration,
        "thresholds": thresholds,
        "operating_points": operating_points,
    }


def _build_sigmoid_calibrator(estimator: object) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=estimator,
        method=CALIBRATION_METHOD,
        cv=build_stratified_cv(),
        ensemble=False,
        n_jobs=-1,
    )


def _predict_malignant_probabilities(estimator: object, X: pd.DataFrame):
    predicted_probabilities = estimator.predict_proba(X)
    malignant_index = list(estimator.classes_).index(MALIGNANT_LABEL)
    return predicted_probabilities[:, malignant_index]


def _validate_probability_inputs(y: pd.Series, probabilities: pd.DataFrame) -> None:
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a Series")

    if not isinstance(probabilities, pd.DataFrame):
        raise ValueError("Probabilities must be a DataFrame")

    if not y.index.equals(probabilities.index):
        raise ValueError("Index mismatch")

    if probabilities.isna().any().any():
        raise ValueError("Null probabilities found")

    if not probabilities.map(lambda value: 0 <= value <= 1).all().all():
        raise ValueError("Probabilities must be between 0 and 1")


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
