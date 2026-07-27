"""HTTP client for the FemHealth FastAPI service."""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class FemHealthApiError(RuntimeError):
    """Safe API error message for presentation layers."""


def resolve_api_base_url() -> str:
    """Resolve the API base URL from the environment."""
    base_url = os.getenv("FEMHEALTH_API_URL", "").strip() or DEFAULT_API_BASE_URL
    return base_url.rstrip("/")


def resolve_api_timeout() -> float:
    """Resolve a positive API timeout from the environment."""
    raw_timeout = os.getenv("FEMHEALTH_API_TIMEOUT_SECONDS", "").strip()

    try:
        timeout = float(raw_timeout)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS

    if not math.isfinite(timeout) or timeout <= 0:
        return DEFAULT_TIMEOUT_SECONDS

    return timeout


def get_health(
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Get API health information."""
    return _request_json("GET", "/health", base_url=base_url, client=client)


def get_model_info(
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Get safe model metadata from the API."""
    return _request_json("GET", "/model", base_url=base_url, client=client)


def get_explainability(
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Get persisted explainability metadata from the API."""
    return _request_json("GET", "/explainability", base_url=base_url, client=client)


def get_explainability_plot(
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> bytes:
    """Get persisted explainability PNG bytes from the API."""
    return _request_bytes("GET", "/explainability/plot", base_url=base_url, client=client)


def get_demo_cases(
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Get validated demonstration cases from the API."""
    return _request_json("GET", "/demo-cases", base_url=base_url, client=client)


def request_prediction(
    features: dict[str, float],
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Request one academic classification from the API."""
    return _request_json(
        "POST",
        "/predict",
        base_url=base_url,
        client=client,
        json_payload={"features": dict(features)},
    )


def _request_json(
    method: str,
    path: str,
    base_url: str | None = None,
    client: httpx.Client | None = None,
    json_payload: dict[str, Any] | None = None,
) -> dict:
    resolved_base_url = resolve_api_base_url() if base_url is None else base_url.rstrip("/")
    timeout = resolve_api_timeout()
    url = f"{resolved_base_url}{path}"

    try:
        if client is None:
            with httpx.Client(timeout=timeout) as created_client:
                response = _send_request(created_client, method, url, timeout, json_payload)
        else:
            response = _send_request(client, method, url, timeout, json_payload)

        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise FemHealthApiError("A API demorou para responder.") from exc
    except httpx.HTTPStatusError as exc:
        raise _http_error(exc.response.status_code) from exc
    except httpx.RequestError as exc:
        raise FemHealthApiError(
            "Não foi possível conectar à API. Verifique se o serviço FastAPI está em execução."
        ) from exc
    except ValueError as exc:
        raise FemHealthApiError("A API retornou uma resposta inválida.") from exc

    if not isinstance(payload, dict):
        raise FemHealthApiError("A API retornou uma resposta inválida.")

    return payload


def _request_bytes(
    method: str,
    path: str,
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> bytes:
    resolved_base_url = resolve_api_base_url() if base_url is None else base_url.rstrip("/")
    timeout = resolve_api_timeout()
    url = f"{resolved_base_url}{path}"

    try:
        if client is None:
            with httpx.Client(timeout=timeout) as created_client:
                response = _send_request(created_client, method, url, timeout, None)
        else:
            response = _send_request(client, method, url, timeout, None)

        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise FemHealthApiError("A API demorou para responder.") from exc
    except httpx.HTTPStatusError as exc:
        raise _http_error(exc.response.status_code) from exc
    except httpx.RequestError as exc:
        raise FemHealthApiError(
            "Não foi possível conectar à API. Verifique se o serviço FastAPI está em execução."
        ) from exc

    content_type = response.headers.get("content-type", "")
    content = response.content
    if "image/png" not in content_type or not content or not content.startswith(PNG_SIGNATURE):
        raise FemHealthApiError("A API retornou uma resposta inválida.")

    return content


def _send_request(
    client: httpx.Client,
    method: str,
    url: str,
    timeout: float,
    json_payload: dict[str, Any] | None,
) -> httpx.Response:
    if method == "GET":
        return client.get(url, timeout=timeout)

    return client.post(url, json=json_payload, timeout=timeout)


def _http_error(status_code: int) -> FemHealthApiError:
    if status_code == 422:
        return FemHealthApiError("A API rejeitou os valores enviados.")

    if status_code == 503:
        return FemHealthApiError("O modelo está temporariamente indisponível.")

    return FemHealthApiError("A API retornou um erro inesperado.")
