import importlib

import streamlit as st

st.set_page_config(
    page_title="FemHealth Clinical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

streamlit_pages = importlib.import_module("femhealth.streamlit_pages")


pages = {
    "Projeto": [
        st.Page(
            streamlit_pages.render_presentation_page,
            title="Apresentação",
            url_path="",
            default=True,
        ),
        st.Page(
            streamlit_pages.render_model_page,
            title="Modelo e resultados",
            url_path="modelo",
        ),
        st.Page(
            streamlit_pages.render_explainability_page,
            title="Explicabilidade",
            url_path="explicabilidade",
        ),
    ],
    "Uso acadêmico": [
        st.Page(
            streamlit_pages.render_demo_cases_page,
            title="Casos de demonstração",
            url_path="casos-demonstracao",
        ),
        st.Page(
            streamlit_pages.render_simulator_page,
            title="Simulador",
            url_path="simulador",
        ),
    ],
}

navigation = st.navigation(pages)
st.sidebar.caption("Projeto acadêmico — sem validade clínica.")
navigation.run()
