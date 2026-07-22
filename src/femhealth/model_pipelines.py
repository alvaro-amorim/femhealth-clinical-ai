"""Candidate model pipelines for future experiments."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Build fresh Scikit-learn pipelines for candidate classifiers."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(random_state=RANDOM_STATE)),
            ]
        ),
    }
