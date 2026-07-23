"""Controlled hyperparameter tuning for selected candidate models."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GridSearchCV

from femhealth.model_evaluation import (
    METRIC_NAMES,
    build_evaluation_scorers,
    build_stratified_cv,
    validate_development_data,
)
from femhealth.model_pipelines import build_candidate_pipelines

TUNED_CANDIDATES = (
    "logistic_regression",
    "random_forest",
    "svm",
)
REFIT_METRIC = "recall_malignant"
TUNED_PARAMETERS = {
    "logistic_regression": {
        "model__C": 0.1,
        "model__class_weight": "balanced",
        "model__l1_ratio": 1.0,
        "model__solver": "liblinear",
    },
    "random_forest": {
        "model__class_weight": "balanced",
        "model__max_depth": None,
        "model__max_features": "sqrt",
        "model__min_samples_leaf": 1,
        "model__n_estimators": 200,
    },
    "svm": {
        "model__C": 1.0,
        "model__class_weight": "balanced",
        "model__gamma": "scale",
        "model__kernel": "rbf",
    },
}

ParameterGrid = dict[str, list[object]]
ParameterGrids = dict[str, ParameterGrid | list[ParameterGrid]]


def build_parameter_grids() -> ParameterGrids:
    """Build the fixed hyperparameter grids for selected candidates."""
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__l1_ratio": [1.0, 0.0],
            "model__solver": ["liblinear"],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [None, "balanced"],
        },
        "svm": [
            {
                "model__kernel": ["linear"],
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__class_weight": [None, "balanced"],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.001, 0.01, 0.1],
                "model__class_weight": [None, "balanced"],
            },
        ],
    }


def build_grid_searches(parameter_grids: ParameterGrids | None = None) -> dict[str, GridSearchCV]:
    """Build fresh grid searches for the selected candidates."""
    pipelines = build_candidate_pipelines()
    grids = build_parameter_grids() if parameter_grids is None else parameter_grids

    return {
        candidate: GridSearchCV(
            estimator=pipelines[candidate],
            param_grid=grids[candidate],
            scoring=build_evaluation_scorers(),
            refit=REFIT_METRIC,
            cv=build_stratified_cv(),
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        for candidate in TUNED_CANDIDATES
    }


def build_tuned_candidate_pipelines():
    """Build fresh unfitted pipelines with selected hyperparameters applied."""
    pipelines = build_candidate_pipelines()

    return {
        candidate: pipelines[candidate].set_params(**TUNED_PARAMETERS[candidate])
        for candidate in TUNED_CANDIDATES
    }


def tune_selected_candidates(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    parameter_grids: ParameterGrids | None = None,
) -> tuple[pd.DataFrame, dict[str, GridSearchCV]]:
    """Tune selected candidates using only development data."""
    validate_development_data(X_development, y_development)

    searches = build_grid_searches(parameter_grids)
    rows = []

    for model_name in TUNED_CANDIDATES:
        search = searches[model_name]
        search.fit(X_development, y_development)
        rows.append(_build_summary_row(model_name, search))

    return pd.DataFrame(rows), searches


def _build_summary_row(model_name: str, search: GridSearchCV) -> dict[str, object]:
    best_index = search.best_index_
    row: dict[str, object] = {
        "model": model_name,
        "refit_metric": REFIT_METRIC,
        "best_params": search.best_params_,
    }

    for metric_name in METRIC_NAMES:
        row[f"{metric_name}_mean"] = search.cv_results_[f"mean_test_{metric_name}"][best_index]
        row[f"{metric_name}_std"] = search.cv_results_[f"std_test_{metric_name}"][best_index]

    return row
