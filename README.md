# FemHealth ML Triage

Projeto academico para estruturar uma base inicial de triagem em saude feminina com
apoio de Machine Learning.

O objetivo geral e criar, em etapas futuras, um fluxo reprodutivel que va de um
notebook de treinamento para um modelo serializado em Joblib, consumido por uma API
FastAPI e apresentado em uma interface Streamlit.

Estado atual: fundacao inicial do projeto, contendo apenas configuracao de pacote,
qualidade de codigo, testes e integracao continua.

Este projeto requer Python 3.11.

## Instalar

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Validar

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Este projeto e academico e nao e uma ferramenta clinica validada.
