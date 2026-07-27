"""FastAPI application for academic inference with the final artifact."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status

from femhealth.api_schemas import (
    ACADEMIC_DISCLAIMER,
    DemoCasesResponse,
    ExplainabilityResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from femhealth.demo_cases_artifact import load_demo_cases_artifact
from femhealth.explainability_artifacts import load_explainability_artifacts
from femhealth.inference import predict_with_artifact
from femhealth.model_artifact import load_model_artifact

API_TITLE = "FemHealth Clinical AI"
API_VERSION = "1.0.0"


def create_app(
    artifact_loader: Callable[[], tuple[object, dict]] | None = None,
    explainability_loader: Callable[[], tuple[dict, bytes]] | None = None,
    demo_cases_loader: Callable[[], dict] | None = None,
) -> FastAPI:
    """Create the FastAPI app without loading the model until lifespan startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = load_model_artifact if artifact_loader is None else artifact_loader
        estimator, metadata = loader()
        explainability_artifact_loader = (
            load_explainability_artifacts
            if explainability_loader is None
            else explainability_loader
        )
        explainability_payload, explainability_plot_bytes = explainability_artifact_loader()
        demo_artifact_loader = (
            load_demo_cases_artifact if demo_cases_loader is None else demo_cases_loader
        )
        demo_cases_payload = demo_artifact_loader()
        app.state.estimator = estimator
        app.state.metadata = metadata
        app.state.explainability_payload = explainability_payload
        app.state.explainability_plot_bytes = explainability_plot_bytes
        app.state.demo_cases_payload = demo_cases_payload
        try:
            yield
        finally:
            if hasattr(app.state, "estimator"):
                del app.state.estimator
            if hasattr(app.state, "metadata"):
                del app.state.metadata
            if hasattr(app.state, "explainability_payload"):
                del app.state.explainability_payload
            if hasattr(app.state, "explainability_plot_bytes"):
                del app.state.explainability_plot_bytes
            if hasattr(app.state, "demo_cases_payload"):
                del app.state.demo_cases_payload

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

    @app.get("/explainability", response_model=ExplainabilityResponse)
    def explainability(request: Request) -> ExplainabilityResponse:
        payload = _get_loaded_explainability_payload(request)
        return ExplainabilityResponse(
            **payload,
            plot_endpoint="/explainability/plot",
            disclaimer=ACADEMIC_DISCLAIMER,
        )

    @app.get("/explainability/plot")
    def explainability_plot(request: Request) -> Response:
        plot_bytes = _get_loaded_explainability_plot_bytes(request)
        return Response(content=plot_bytes, media_type="image/png")

    @app.get("/demo-cases", response_model=DemoCasesResponse)
    def demo_cases(request: Request) -> DemoCasesResponse:
        payload = _get_loaded_demo_cases_payload(request)
        return DemoCasesResponse(**payload, disclaimer=ACADEMIC_DISCLAIMER)

    return app


def _get_loaded_artifact(request: Request) -> tuple[object, dict]:
    if not hasattr(request.app.state, "estimator") or not hasattr(request.app.state, "metadata"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable",
        )

    return request.app.state.estimator, request.app.state.metadata


def _get_loaded_explainability_payload(request: Request) -> dict:
    if not hasattr(request.app.state, "explainability_payload"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability artifact is unavailable",
        )

    return request.app.state.explainability_payload


def _get_loaded_explainability_plot_bytes(request: Request) -> bytes:
    if not hasattr(request.app.state, "explainability_plot_bytes"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability artifact is unavailable",
        )

    return request.app.state.explainability_plot_bytes


def _get_loaded_demo_cases_payload(request: Request) -> dict:
    if not hasattr(request.app.state, "demo_cases_payload"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo cases artifact is unavailable",
        )

    return request.app.state.demo_cases_payload


app = create_app()
