"""FastAPI application for academic inference with the final artifact."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status

from femhealth.api_schemas import (
    ACADEMIC_DISCLAIMER,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from femhealth.inference import predict_with_artifact
from femhealth.model_artifact import load_model_artifact

API_TITLE = "FemHealth Clinical AI"
API_VERSION = "1.0.0"


def create_app(
    artifact_loader: Callable[[], tuple[object, dict]] | None = None,
) -> FastAPI:
    """Create the FastAPI app without loading the model until lifespan startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = load_model_artifact if artifact_loader is None else artifact_loader
        estimator, metadata = loader()
        app.state.estimator = estimator
        app.state.metadata = metadata
        try:
            yield
        finally:
            if hasattr(app.state, "estimator"):
                del app.state.estimator
            if hasattr(app.state, "metadata"):
                del app.state.metadata

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        estimator, metadata = _get_loaded_artifact(request)
        return HealthResponse(
            status="ok",
            model_loaded=estimator is not None,
            artifact_version=metadata["artifact_version"],
            selected_variant=metadata["selected_variant"],
        )

    @app.get("/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        _, metadata = _get_loaded_artifact(request)
        return ModelInfoResponse(
            artifact_version=metadata["artifact_version"],
            selected_variant=metadata["selected_variant"],
            selected_model=metadata["selected_model"],
            selected_calibration=metadata["selected_calibration"],
            threshold=metadata["threshold"],
            class_labels=metadata["class_labels"],
            feature_count=metadata["feature_count"],
            feature_names=metadata["feature_names"],
            training_sample_count=metadata["training_sample_count"],
            model_sha256=metadata["model_sha256"],
            final_holdout_metrics=metadata["final_holdout_metrics"],
            disclaimer=ACADEMIC_DISCLAIMER,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request_body: PredictionRequest, request: Request) -> PredictionResponse:
        estimator, metadata = _get_loaded_artifact(request)
        feature_names = metadata["feature_names"]
        X = pd.DataFrame(
            [[request_body.features[feature_name] for feature_name in feature_names]],
            index=["request"],
            columns=feature_names,
        )
        predictions = predict_with_artifact(X, estimator, metadata)
        prediction = predictions.iloc[0]

        return PredictionResponse(
            probability_malignant=float(prediction["probability_malignant"]),
            probability_benign=float(prediction["probability_benign"]),
            predicted_label=int(prediction["predicted_label"]),
            predicted_class=str(prediction["predicted_class"]),
            threshold=float(metadata["threshold"]),
            disclaimer=ACADEMIC_DISCLAIMER,
        )

    return app


def _get_loaded_artifact(request: Request) -> tuple[object, dict]:
    if not hasattr(request.app.state, "estimator") or not hasattr(request.app.state, "metadata"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable",
        )

    return request.app.state.estimator, request.app.state.metadata


app = create_app()
