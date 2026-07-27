import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import femhealth.model_explainability as explainability
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.final_evaluation import DEVELOPMENT_CLASS_DISTRIBUTION, DEVELOPMENT_SAMPLE_COUNT
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL, build_stratified_cv
from femhealth.model_explainability import (
    EXPLAINABILITY_CV_SPLITS,
    EXPLAINABILITY_RANDOM_STATE,
    EXPLAINABILITY_SCORER,
    PERMUTATION_REPEATS,
    compute_cross_validated_permutation_importance,
    get_malignant_probabilities,
    score_malignant_roc_auc,
)


class FixedProbabilityEstimator:
    classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X):
        return np.asarray(self.probabilities[: len(X)])


class RecordingEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, records):
        self.records = records
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.records.append({"train_index": X.index.copy(), "target_index": y.index.copy()})
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.pipeline.predict_proba(X)


class SignalEstimator:
    classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict_proba(self, X: pd.DataFrame):
        malignant_probability = 1 / (1 + np.exp(-X["mean radius"].to_numpy()))
        return np.column_stack([1 - malignant_probability, malignant_probability])


@pytest.fixture(scope="module")
def synthetic_development_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    y = pd.Series(
        [MALIGNANT_LABEL] * DEVELOPMENT_CLASS_DISTRIBUTION[MALIGNANT_LABEL]
        + [BENIGN_LABEL] * DEVELOPMENT_CLASS_DISTRIBUTION[BENIGN_LABEL],
        index=pd.Index(range(DEVELOPMENT_SAMPLE_COUNT), name="sample_index"),
        name="diagnosis",
    )
    X = pd.DataFrame(
        rng.normal(0, 1, size=(DEVELOPMENT_SAMPLE_COUNT, len(WDBC_FEATURE_NAMES))),
        index=y.index,
        columns=WDBC_FEATURE_NAMES,
    )
    malignant_signal = y.eq(MALIGNANT_LABEL).astype(float)
    X["mean radius"] = malignant_signal * 4.0 + rng.normal(0, 0.05, size=len(X))
    return X, y


@pytest.fixture(scope="module")
def explainability_result(synthetic_development_data):
    X, y = synthetic_development_data
    records: list[dict[str, pd.Index]] = []

    def estimator_factory():
        return RecordingEstimator(records)

    details, summary, fold_scores = compute_cross_validated_permutation_importance(
        X,
        y,
        estimator_factory=estimator_factory,
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )
    return details, summary, fold_scores, records


def test_explainability_constants_are_expected() -> None:
    assert EXPLAINABILITY_RANDOM_STATE == 42
    assert EXPLAINABILITY_CV_SPLITS == 5
    assert PERMUTATION_REPEATS == 10
    assert EXPLAINABILITY_SCORER == "roc_auc_malignant"


def test_compute_rejects_unexpected_cv_split_count(
    monkeypatch,
    synthetic_development_data,
) -> None:
    X, y = synthetic_development_data

    class UnexpectedSplitCountCv:
        def get_n_splits(self):
            return 4

    monkeypatch.setattr(
        explainability,
        "build_stratified_cv",
        lambda: UnexpectedSplitCountCv(),
    )

    with pytest.raises(RuntimeError, match="Unexpected explainability CV split count"):
        compute_cross_validated_permutation_importance(
            X,
            y,
            estimator_factory=lambda: SignalEstimator(),
            n_repeats=1,
            n_jobs=1,
        )


def test_get_malignant_probabilities_uses_explicit_class_column() -> None:
    estimator = FixedProbabilityEstimator([[0.80, 0.20], [0.25, 0.75]])
    X = pd.DataFrame({"x": [1.0, 2.0]})

    probabilities = get_malignant_probabilities(estimator, X)

    assert probabilities.tolist() == [0.20, 0.75]


def test_get_malignant_probabilities_rejects_missing_malignant_class() -> None:
    estimator = FixedProbabilityEstimator([[1.0], [1.0]])
    estimator.classes_ = [BENIGN_LABEL]

    with pytest.raises(ValueError, match="malignant class"):
        get_malignant_probabilities(estimator, pd.DataFrame({"x": [1.0, 2.0]}))


def test_score_malignant_roc_auc_uses_malignant_target() -> None:
    estimator = FixedProbabilityEstimator([[0.90, 0.10], [0.20, 0.80], [0.80, 0.20]])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([BENIGN_LABEL, MALIGNANT_LABEL, BENIGN_LABEL])

    assert score_malignant_roc_auc(estimator, X, y) == 1.0


def test_compute_outputs_have_expected_shapes(explainability_result) -> None:
    details, summary, fold_scores, _ = explainability_result

    assert len(details) == 5 * 30 * 2
    assert len(summary) == 30
    assert len(fold_scores) == 5
    assert list(fold_scores["fold"]) == [1, 2, 3, 4, 5]


def test_each_fold_uses_fresh_estimator_and_disjoint_validation(
    synthetic_development_data,
    explainability_result,
) -> None:
    X, y = synthetic_development_data
    _, _, _, records = explainability_result

    assert len(records) == 5
    for record, (_, validation_positions) in zip(
        records,
        build_stratified_cv().split(X, y),
        strict=True,
    ):
        validation_index = X.iloc[validation_positions].index
        assert record["train_index"].equals(record["target_index"])
        assert set(record["train_index"]).isdisjoint(set(validation_index))


def test_summary_ranking_and_counts_are_valid(explainability_result) -> None:
    _, summary, _, _ = explainability_result

    assert summary["rank"].tolist() == list(range(1, 31))
    assert summary["mean_importance"].is_monotonic_decreasing
    assert set(summary["feature_position"]) == set(range(1, 31))
    assert summary["observation_count"].eq(10).all()
    assert summary["fold_count"].eq(5).all()
    assert summary["positive_fraction"].between(0, 1).all()


def test_signal_feature_is_among_top_five(explainability_result) -> None:
    _, summary, _, _ = explainability_result

    assert "mean radius" in summary.head(5)["feature_name"].to_list()


def test_results_are_deterministic_with_same_seed(synthetic_development_data) -> None:
    X, y = synthetic_development_data

    first = compute_cross_validated_permutation_importance(
        X,
        y,
        estimator_factory=lambda: SignalEstimator(),
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )
    second = compute_cross_validated_permutation_importance(
        X,
        y,
        estimator_factory=lambda: SignalEstimator(),
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )

    for first_frame, second_frame in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_frame, second_frame)


def test_inputs_are_not_modified(synthetic_development_data) -> None:
    X, y = synthetic_development_data
    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    compute_cross_validated_permutation_importance(
        X,
        y,
        estimator_factory=lambda: SignalEstimator(),
        n_repeats=1,
        n_jobs=1,
    )

    assert X.equals(original_X)
    assert y.equals(original_y)


def test_invalid_inputs_are_rejected(synthetic_development_data) -> None:
    X, y = synthetic_development_data

    with pytest.raises(ValueError, match="Unexpected development sample count"):
        compute_cross_validated_permutation_importance(
            X.iloc[:-1],
            y.iloc[:-1],
            estimator_factory=lambda: SignalEstimator(),
        )

    invalid_y = y.copy()
    invalid_y.iloc[0] = BENIGN_LABEL
    with pytest.raises(ValueError, match="Unexpected development class distribution"):
        compute_cross_validated_permutation_importance(
            X,
            invalid_y,
            estimator_factory=lambda: SignalEstimator(),
        )

    with pytest.raises(ValueError, match="Unexpected feature count"):
        compute_cross_validated_permutation_importance(
            X.drop(columns=[WDBC_FEATURE_NAMES[0]]),
            y,
            estimator_factory=lambda: SignalEstimator(),
        )

    with pytest.raises(ValueError, match="Unexpected feature columns"):
        compute_cross_validated_permutation_importance(
            X[list(reversed(WDBC_FEATURE_NAMES))],
            y,
            estimator_factory=lambda: SignalEstimator(),
        )

    with pytest.raises(ValueError, match="n_repeats"):
        compute_cross_validated_permutation_importance(
            X,
            y,
            estimator_factory=lambda: SignalEstimator(),
            n_repeats=0,
        )


def test_negative_importances_are_preserved(monkeypatch, synthetic_development_data) -> None:
    X, y = synthetic_development_data

    class FakePermutationResult:
        importances = np.full((len(WDBC_FEATURE_NAMES), 1), -0.01)

    monkeypatch.setattr(
        explainability,
        "permutation_importance",
        lambda *args, **kwargs: FakePermutationResult(),
    )

    details, summary, _ = compute_cross_validated_permutation_importance(
        X,
        y,
        estimator_factory=lambda: SignalEstimator(),
        n_repeats=1,
        n_jobs=1,
    )

    assert details["importance"].lt(0).all()
    assert summary["min_importance"].lt(0).all()


def test_model_explainability_module_has_no_forbidden_operations() -> None:
    source = inspect.getsource(explainability)

    assert "load_wdbc_data" not in source
    assert "split_development_test" not in source
    assert "evaluate_final_holdout" not in source
    assert "GridSearchCV" not in source
    assert "SelectKBest" not in source
    assert "feature_selection" not in source
    assert "SELECTED_THRESHOLD" not in source
