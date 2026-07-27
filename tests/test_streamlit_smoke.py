from pathlib import Path

from streamlit.testing.v1 import AppTest

DEMO_PAGE_APPTEST_SCRIPT = r'''
import streamlit as st

from femhealth import streamlit_pages
from femhealth.demo_cases_artifact import load_demo_cases_artifact

payload = load_demo_cases_artifact()
st.session_state.setdefault("prediction_calls", 0)


def fake_get_demo_cases():
    return payload


def fake_request_prediction(features):
    st.session_state["prediction_calls"] += 1
    mean_radius = features["mean radius"]
    if mean_radius == payload["cases"][3]["features"]["mean radius"]:
        return {
            "probability_malignant": 0.62,
            "probability_benign": 0.38,
            "predicted_label": 0,
            "predicted_class": "malignant",
            "threshold": 0.51,
        }

    return {
        "probability_malignant": 0.99,
        "probability_benign": 0.01,
        "predicted_label": 0,
        "predicted_class": "malignant",
        "threshold": 0.51,
    }


streamlit_pages.get_demo_cases = fake_get_demo_cases
streamlit_pages.request_prediction = fake_request_prediction
streamlit_pages.render_demo_cases_page()
'''


def test_streamlit_app_renders_with_unavailable_api(monkeypatch) -> None:
    monkeypatch.setenv("FEMHEALTH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("FEMHEALTH_API_TIMEOUT_SECONDS", "0.05")

    app = AppTest.from_file("streamlit_app.py").run(timeout=30)

    assert not app.exception
    assert any("FemHealth Clinical AI" in title.value for title in app.title)
    assert any(
        "Não foi possível conectar à API" in warning.value
        or "A API demorou para responder" in warning.value
        for warning in app.warning
    )
    assert "render_demo_cases_page" in Path("streamlit_app.py").read_text(encoding="utf-8")


def test_demo_cases_page_restores_progress_from_query_params_without_inference() -> None:
    app = AppTest.from_string(DEMO_PAGE_APPTEST_SCRIPT, default_timeout=10).run(timeout=10)

    app.button[0].click().run(timeout=10)
    app.selectbox[0].select_index(3).run(timeout=10)
    app.button[0].click().run(timeout=10)

    assert not app.exception
    assert _scoreboard_values(app) == ["2", "1", "1", "50,00%"]
    assert app.session_state["prediction_calls"] == 2
    encoded_progress = app.query_params["demo_progress"][0]

    restored_app = AppTest.from_string(DEMO_PAGE_APPTEST_SCRIPT, default_timeout=10)
    restored_app.query_params["demo_progress"] = [encoded_progress]
    restored_app.query_params["other"] = ["keep"]
    restored_app.run(timeout=10)

    assert not restored_app.exception
    assert _scoreboard_values(restored_app) == ["2", "1", "1", "50,00%"]
    assert restored_app.session_state["prediction_calls"] == 0
    assert restored_app.selectbox[0].value[1]["case_id"] == "demo-04"

    restored_app.button[1].click().run(timeout=10)

    assert _scoreboard_values(restored_app) == ["0", "0", "0", "—"]
    assert "demo_progress" not in restored_app.query_params
    assert restored_app.query_params == {"other": ["keep"]}


def test_demo_cases_page_ignores_invalid_query_progress_without_inference() -> None:
    app = AppTest.from_string(DEMO_PAGE_APPTEST_SCRIPT, default_timeout=10)
    app.query_params["demo_progress"] = ["!!!!"]
    app.query_params["other"] = ["keep"]
    app.run(timeout=10)

    assert not app.exception
    assert _scoreboard_values(app) == ["0", "0", "0", "—"]
    assert app.session_state["prediction_calls"] == 0
    assert "demo_progress" not in app.query_params
    assert app.query_params == {"other": ["keep"]}


def _scoreboard_values(app: AppTest) -> list[str]:
    return [metric.value for metric in app.metric[-4:]]
