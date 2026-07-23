from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from femhealth.final_selection import (
    SELECTED_CALIBRATION,
    SELECTED_MODEL,
    SELECTED_THRESHOLD,
    SELECTED_VARIANT,
    build_selected_estimator,
)


def test_final_selection_constants_are_frozen() -> None:
    assert SELECTED_VARIANT == "svm_sigmoid"
    assert SELECTED_MODEL == "svm"
    assert SELECTED_THRESHOLD == 0.51
    assert SELECTED_CALIBRATION == "sigmoid"


def test_build_selected_estimator_returns_sigmoid_calibrator() -> None:
    estimator = build_selected_estimator()

    assert isinstance(estimator, CalibratedClassifierCV)
    assert estimator.method == "sigmoid"
    assert estimator.ensemble is False
    assert estimator.cv.n_splits == 5
    assert estimator.cv.shuffle is True
    assert estimator.cv.random_state == 42


def test_selected_estimator_wraps_tuned_svm_pipeline() -> None:
    estimator = build_selected_estimator()
    pipeline = estimator.estimator

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], SVC)
    assert pipeline.named_steps["model"].kernel == "rbf"
    assert pipeline.named_steps["model"].C == 1.0
    assert pipeline.named_steps["model"].gamma == "scale"
    assert pipeline.named_steps["model"].class_weight == "balanced"
    assert pipeline.named_steps["model"].probability is not True


def test_build_selected_estimator_returns_independent_unfitted_objects() -> None:
    first = build_selected_estimator()
    second = build_selected_estimator()

    assert first is not second
    assert first.estimator is not second.estimator
    assert first.estimator.named_steps["model"] is not second.estimator.named_steps["model"]
    assert not hasattr(first, "classes_")
    assert not hasattr(first.estimator.named_steps["model"], "classes_")
