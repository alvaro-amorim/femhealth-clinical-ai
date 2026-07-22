import inspect

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import femhealth.model_pipelines as model_pipelines_module
from femhealth.model_pipelines import RANDOM_STATE, build_candidate_pipelines

EXPECTED_CANDIDATES = {
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
    "svm",
}


def test_build_candidate_pipelines_has_exact_candidates() -> None:
    pipelines = build_candidate_pipelines()

    assert set(pipelines) == EXPECTED_CANDIDATES


def test_all_candidates_are_pipelines() -> None:
    pipelines = build_candidate_pipelines()

    assert all(isinstance(pipeline, Pipeline) for pipeline in pipelines.values())


def test_linear_distance_and_margin_models_have_scaler() -> None:
    pipelines = build_candidate_pipelines()

    for candidate in ("logistic_regression", "knn", "svm"):
        assert isinstance(pipelines[candidate].named_steps["scaler"], StandardScaler)


def test_tree_based_models_do_not_have_scaler() -> None:
    pipelines = build_candidate_pipelines()

    for candidate in ("decision_tree", "random_forest"):
        assert list(pipelines[candidate].named_steps) == ["model"]


def test_each_pipeline_ends_with_expected_estimator() -> None:
    pipelines = build_candidate_pipelines()

    assert isinstance(pipelines["logistic_regression"].steps[-1][1], LogisticRegression)
    assert isinstance(pipelines["knn"].steps[-1][1], KNeighborsClassifier)
    assert isinstance(pipelines["decision_tree"].steps[-1][1], DecisionTreeClassifier)
    assert isinstance(pipelines["random_forest"].steps[-1][1], RandomForestClassifier)
    assert isinstance(pipelines["svm"].steps[-1][1], SVC)


def test_random_state_is_set_on_estimators_that_support_it() -> None:
    pipelines = build_candidate_pipelines()

    assert pipelines["logistic_regression"].named_steps["model"].random_state == RANDOM_STATE
    assert pipelines["decision_tree"].named_steps["model"].random_state == RANDOM_STATE
    assert pipelines["random_forest"].named_steps["model"].random_state == RANDOM_STATE
    assert pipelines["svm"].named_steps["model"].random_state == RANDOM_STATE


def test_build_candidate_pipelines_returns_independent_objects() -> None:
    first_pipelines = build_candidate_pipelines()
    second_pipelines = build_candidate_pipelines()

    for candidate in EXPECTED_CANDIDATES:
        assert first_pipelines[candidate] is not second_pipelines[candidate]
        assert first_pipelines[candidate].named_steps["model"] is not (
            second_pipelines[candidate].named_steps["model"]
        )

        if "scaler" in first_pipelines[candidate].named_steps:
            assert first_pipelines[candidate].named_steps["scaler"] is not (
                second_pipelines[candidate].named_steps["scaler"]
            )


def test_pipelines_are_not_fitted_when_created() -> None:
    pipelines = build_candidate_pipelines()

    for pipeline in pipelines.values():
        assert not hasattr(pipeline, "n_features_in_")
        assert not hasattr(pipeline.named_steps["model"], "classes_")

        if "scaler" in pipeline.named_steps:
            assert not hasattr(pipeline.named_steps["scaler"], "mean_")


def test_model_pipelines_do_not_access_final_test_split() -> None:
    module_source = inspect.getsource(model_pipelines_module)

    assert "split_development_test" not in module_source
    assert "data_split" not in module_source
    assert "X_test" not in module_source
    assert "y_test" not in module_source
