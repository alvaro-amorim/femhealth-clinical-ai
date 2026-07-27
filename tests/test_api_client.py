import httpx
import pytest

from femhealth.api_client import (
    DEFAULT_API_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    PNG_SIGNATURE,
    FemHealthApiError,
    get_explainability,
    get_explainability_plot,
    get_health,
    get_model_info,
    request_prediction,
    resolve_api_base_url,
    resolve_api_timeout,
)


def test_resolve_api_base_url_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("FEMHEALTH_API_URL", raising=False)

    assert resolve_api_base_url() == DEFAULT_API_BASE_URL


def test_resolve_api_base_url_uses_env_and_removes_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("FEMHEALTH_API_URL", "http://localhost:9000/")

    assert resolve_api_base_url() == "http://localhost:9000"


def test_resolve_api_timeout_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("FEMHEALTH_API_TIMEOUT_SECONDS", raising=False)

    assert resolve_api_timeout() == DEFAULT_TIMEOUT_SECONDS


def test_resolve_api_timeout_uses_positive_env(monkeypatch) -> None:
    monkeypatch.setenv("FEMHEALTH_API_TIMEOUT_SECONDS", "1.5")

    assert resolve_api_timeout() == 1.5


@pytest.mark.parametrize("raw_timeout", ["invalid", "0", "-1", "nan", "inf", "-inf"])
def test_resolve_api_timeout_invalid_values_use_default(monkeypatch, raw_timeout) -> None:
    monkeypatch.setenv("FEMHEALTH_API_TIMEOUT_SECONDS", raw_timeout)

    assert resolve_api_timeout() == DEFAULT_TIMEOUT_SECONDS


def test_get_health_makes_expected_request() -> None:
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={"status": "ok"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = get_health(client=client)

    assert payload == {"status": "ok"}
    assert len(seen_requests) == 1
    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/health"


def test_get_model_info_makes_expected_request() -> None:
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={"selected_model": "svm"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = get_model_info(client=client)

    assert payload == {"selected_model": "svm"}
    assert len(seen_requests) == 1
    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/model"


def test_request_prediction_makes_expected_request_without_mutating_payload() -> None:
    seen_requests = []
    features = {"mean radius": 1.0}
    original_features = features.copy()

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={"predicted_class": "benign"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = request_prediction(features, client=client)

    assert payload == {"predicted_class": "benign"}
    assert features == original_features
    assert len(seen_requests) == 1
    assert seen_requests[0].method == "POST"
    assert seen_requests[0].url.path == "/predict"
    assert seen_requests[0].read() == b'{"features":{"mean radius":1.0}}'


def test_get_explainability_makes_expected_request() -> None:
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={"feature_count": 30}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = get_explainability(client=client)

    assert payload == {"feature_count": 30}
    assert len(seen_requests) == 1
    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/explainability"


def test_get_explainability_plot_makes_expected_request_and_returns_png() -> None:
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            content=PNG_SIGNATURE + b"plot",
            headers={"content-type": "image/png"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = get_explainability_plot(client=client)

    assert payload == PNG_SIGNATURE + b"plot"
    assert len(seen_requests) == 1
    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/explainability/plot"


@pytest.mark.parametrize(
    ("headers", "content"),
    [
        ({"content-type": "application/json"}, PNG_SIGNATURE + b"plot"),
        ({"content-type": "image/png"}, b""),
        ({"content-type": "image/png"}, b"not-png"),
    ],
)
def test_get_explainability_plot_rejects_invalid_png(headers, content) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers=headers, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match="resposta inv"):
            get_explainability_plot(client=client)


def test_timeout_uses_safe_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match="A API demorou para responder."):
            get_health(client=client)


def test_explainability_plot_timeout_uses_safe_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match="demorou"):
            get_explainability_plot(client=client)


def test_connection_error_uses_safe_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match="Não foi possível conectar à API"):
            get_health(client=client)


def test_explainability_plot_connection_error_uses_safe_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match="conectar"):
            get_explainability_plot(client=client)


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (422, "A API rejeitou os valores enviados."),
        (503, "O modelo está temporariamente indisponível."),
        (500, "A API retornou um erro inesperado."),
    ],
)
def test_http_errors_use_safe_messages(status_code, message) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "internal"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match=message):
            get_health(client=client)


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (503, "temporariamente"),
        (500, "inesperado"),
    ],
)
def test_explainability_plot_http_errors_use_safe_messages(status_code, message) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "internal"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match=message):
            get_explainability_plot(client=client)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=[]),
        httpx.Response(200, content=b"not-json"),
    ],
)
def test_invalid_json_payload_is_rejected(response) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FemHealthApiError, match="A API retornou uma resposta inválida."):
            get_health(client=client)


def test_injected_client_is_not_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        get_health(client=client)
        assert not client.is_closed
    finally:
        client.close()


def test_injected_client_for_plot_is_not_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=PNG_SIGNATURE + b"plot",
            headers={"content-type": "image/png"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        get_explainability_plot(client=client)
        assert not client.is_closed
    finally:
        client.close()


def test_internally_created_client_is_closed(monkeypatch) -> None:
    instances = []

    class FakeClient:
        def __init__(self, timeout: float):
            self.timeout = timeout
            self.closed = False
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.closed = True

        def get(self, url: str, timeout: float) -> httpx.Response:
            request = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok"}, request=request)

    monkeypatch.setattr("femhealth.api_client.httpx.Client", FakeClient)

    assert get_health() == {"status": "ok"}
    assert len(instances) == 1
    assert instances[0].closed is True


def test_internally_created_client_for_plot_is_closed(monkeypatch) -> None:
    instances = []

    class FakeClient:
        def __init__(self, timeout: float):
            self.timeout = timeout
            self.closed = False
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.closed = True

        def get(self, url: str, timeout: float) -> httpx.Response:
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                content=PNG_SIGNATURE + b"plot",
                headers={"content-type": "image/png"},
                request=request,
            )

    monkeypatch.setattr("femhealth.api_client.httpx.Client", FakeClient)

    assert get_explainability_plot() == PNG_SIGNATURE + b"plot"
    assert len(instances) == 1
    assert instances[0].closed is True
