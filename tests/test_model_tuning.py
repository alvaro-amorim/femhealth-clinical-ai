import inspect

import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import femhealth.model_tuning as model_tuning_module
from femhealth.data import load_wdbc_data
from femhealth.data_split import split_development_test
from femhealth.model_evaluation import (
    METRIC_NAMES,
    evaluate_baseline_candidates,
)
from femhealth.model_tuning import (
    REFIT_METRIC,
    TUNED_CANDIDATES,
    TUNED_PARAMETERS,
    build_grid_searches,
    build_parameter_grids,
    build_tuned_candidate_pipelines,
    tune_selected_candidates,
)

EXPECTED_SUMMARY_COLUMNS = [
    "model",
    "refit_metric",
    "best_params",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "precision_malignant_mean",
    "precision_malignant_std",
    "recall_malignant_mean",
    "recall_malignant_std",
    "f1_malignant_mean",
    "f1_malignant_std",
    "specificity_benign_mean",
    "specificity_benign_std",
    "roc_auc_malignant_mean",
    "roc_auc_malignant_std",
    "average_precision_malignant_mean",
    "average_precision_malignant_std",
]


@pytest.fixture(scope="module")
def development_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_wdbc_data()
    X_development, _, y_development, _ = split_development_test(X, y)
    return X_development, y_development


@pytest.fixture(scope="module")
def minimal_parameter_grids():
    return {
        "logistic_regression": {
            "model__C": [1.0],
            "model__l1_ratio": [0.0],
            "model__solver": ["liblinear"],
            "model__class_weight": [None],
        },
        "random_forest": {
            "model__n_estimators": [10],
            "model__max_depth": [5],
            "model__min_samples_leaf": [1],
            "model__max_features": ["sqrt"],
            "model__class_weight": [None],
        },
        "svm": [
            {
                "model__kernel": ["linear"],
                "model__C": [1.0],
                "model__class_weight": [None],
            },
        ],
    }


@pytest.fixture(scope="module")
def tuned_results(
    development_data: tuple[pd.DataFrame, pd.Series],
    minimal_parameter_grids,
):
    X_development, y_development = development_data
    return tune_selected_candidates(
        X_development,
        y_development,
        parameter_grids=minimal_parameter_grids,
    )


def test_tuned_candidates_are_exactly_selected_models() -> None:
    assert TUNED_CANDIDATES == ("logistic_regression", "random_forest", "svm")


def test_tuned_parameters_match_selected_round_results() -> None:
    assert TUNED_PARAMETERS == {
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


def test_build_tuned_candidate_pipelines_applies_parameters() -> None:
    pipelines = build_tuned_candidate_pipelines()

    assert tuple(pipelines) == TUNED_CANDIDATES
    for candidate, parameters in TUNED_PARAMETERS.items():
        pipeline_parameters = pipelines[candidate].get_params()
        assert all(pipeline_parameters[name] == value for name, value in parameters.items())


def test_build_tuned_candidate_pipelines_returns_independent_unfitted_objects() -> None:
    first_pipelines = build_tuned_candidate_pipelines()
    second_pipelines = build_tuned_candidate_pipelines()

    for candidate in TUNED_CANDIDATES:
        assert first_pipelines[candidate] is not second_pipelines[candidate]
        assert first_pipelines[candidate].named_steps["model"] is not (
            second_pipelines[candidate].named_steps["model"]
        )
        assert not hasattr(first_pipelines[candidate].named_steps["model"], "classes_")


def test_parameter_grids_use_only_model_prefixed_parameters() -> None:
    for grid in build_parameter_grids().values():
        grid_parts = grid if isinstance(grid, list) else [grid]
        for grid_part in grid_parts:
            assert all(parameter.startswith("model__") for parameter in grid_part)


def test_parameter_grids_are_exact() -> None:
    assert build_parameter_grids() == {
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


def test_svm_has_separate_linear_and_rbf_grids() -> None:
    svm_grid = build_parameter_grids()["svm"]

    assert isinstance(svm_grid, list)
    assert len(svm_grid) == 2
    assert svm_grid[0]["model__kernel"] == ["linear"]
    assert "model__gamma" not in svm_grid[0]
    assert svm_grid[1]["model__kernel"] == ["rbf"]
    assert svm_grid[1]["model__gamma"] == ["scale", 0.001, 0.01, 0.1]


def test_grid_searches_configuration(minimal_parameter_grids) -> None:
    searches = build_grid_searches(parameter_grids=minimal_parameter_grids)

    assert set(searches) == set(TUNED_CANDIDATES)
    for search in searches.values():
        assert isinstance(search, GridSearchCV)
        assert search.refit == REFIT_METRIC
        assert isinstance(search.cv, StratifiedKFold)
        assert search.cv.n_splits == 5
        assert search.cv.shuffle is True
        assert search.cv.random_state == 42
        assert search.error_score == "raise"
        assert set(search.scoring) == set(METRIC_NAMES)


def test_grid_search_pipelines_preserve_scaler_where_applicable(minimal_parameter_grids) -> None:
    searches = build_grid_searches(parameter_grids=minimal_parameter_grids)

    logistic_scaler = searches["logistic_regression"].estimator.named_steps["scaler"]

    assert isinstance(logistic_scaler, StandardScaler)
    assert isinstance(searches["svm"].estimator.named_steps["scaler"], StandardScaler)
    assert "scaler" not in searches["random_forest"].estimator.named_steps


def test_build_grid_searches_returns_independent_objects(minimal_parameter_grids) -> None:
    first_searches = build_grid_searches(parameter_grids=minimal_parameter_grids)
    second_searches = build_grid_searches(parameter_grids=minimal_parameter_grids)

    for candidate in TUNED_CANDIDATES:
        assert first_searches[candidate] is not second_searches[candidate]
        assert first_searches[candidate].estimator is not second_searches[candidate].estimator
        assert first_searches[candidate].estimator.named_steps["model"] is not (
            second_searches[candidate].estimator.named_steps["model"]
        )


def test_tuning_summary_has_expected_columns(tuned_results) -> None:
    summary, _ = tuned_results

    assert list(summary.columns) == EXPECTED_SUMMARY_COLUMNS


def test_tuning_summary_preserves_candidate_order(tuned_results) -> None:
    summary, _ = tuned_results

    assert summary["model"].tolist() == list(TUNED_CANDIDATES)


def test_best_parameters_belong_to_supplied_grids(tuned_results, minimal_parameter_grids) -> None:
    summary, _ = tuned_results

    for row in summary.itertuples(index=False):
        candidate_grid = minimal_parameter_grids[row.model]
        grid_part = candidate_grid[0] if isinstance(candidate_grid, list) else candidate_grid
        assert all(value in grid_part[parameter] for parameter, value in row.best_params.items())


def test_tuning_does_not_modify_development_data(
    development_data: tuple[pd.DataFrame, pd.Series],
    minimal_parameter_grids,
) -> None:
    X_development, y_development = development_data
    original_X = X_development.copy(deep=True)
    original_y = y_development.copy(deep=True)

    tune_selected_candidates(X_development, y_development, parameter_grids=minimal_parameter_grids)

    assert X_development.equals(original_X)
    assert y_development.equals(original_y)


def test_model_tuning_does_not_use_final_holdout_data() -> None:
    module_source = inspect.getsource(model_tuning_module)

    assert "split_development_test" not in module_source
    assert "data_split" not in module_source
    assert "X_test" not in module_source
    assert "y_test" not in module_source


def test_baseline_benchmark_still_produces_expected_structure(
    development_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_development, y_development = development_data
    baseline = evaluate_baseline_candidates(X_development, y_development)

    assert baseline.shape[0] == 5
    assert "best_params" not in baseline.columns
    assert baseline["model"].tolist() == [
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "svm",
    ]
