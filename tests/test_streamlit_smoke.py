from pathlib import Path

from streamlit.testing.v1 import AppTest


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
