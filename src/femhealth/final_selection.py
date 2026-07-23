"""Frozen final model selection made before holdout evaluation."""

from __future__ import annotations

from femhealth.probability_analysis import build_probability_estimators

SELECTED_VARIANT = "svm_sigmoid"
SELECTED_MODEL = "svm"
SELECTED_THRESHOLD = 0.51
SELECTED_CALIBRATION = "sigmoid"


def build_selected_estimator():
    """Build a fresh unfitted estimator for the frozen selected variant."""
    return build_probability_estimators()[SELECTED_VARIANT]
