# FemHealth ML Triage

Projeto academico para estruturar uma base inicial de triagem em saude feminina com
apoio de Machine Learning.

O objetivo geral e criar, em etapas futuras, um fluxo reprodutivel que va de um
notebook de treinamento para um modelo serializado em Joblib, consumido por uma API
FastAPI e apresentado em uma interface Streamlit.

Estado atual: fundacao inicial do projeto com carregamento e contrato inicial do
dataset WDBC, alem de configuracao de pacote, qualidade de codigo, testes e
integracao continua.

## Dataset

O dataset utilizado e o Breast Cancer Wisconsin Diagnostic (WDBC), carregado
exclusivamente por `sklearn.datasets.load_breast_cancer(as_frame=True)`.

Contrato principal:

- 569 amostras;
- 30 variaveis numericas;
- target `diagnosis`, com `0 = malignant` e `1 = benign`;
- 212 casos malignos e 357 casos benignos;
- nenhuma variavel ausente;
- ordem canonica das 30 features preservada.

## Separacao dos dados

O WDBC e separado de forma estratificada em 80% para desenvolvimento e 20% para
teste final. O conjunto de desenvolvimento sera usado futuramente para comparar
abordagens e ajustar decisoes metodologicas. O teste final deve permanecer
preservado para avaliacao apos a selecao.

Nenhum treinamento foi implementado ainda.

## Modelos candidatos

Os candidatos planejados sao Regressao Logistica, KNN, Arvore de Decisao,
Random Forest e SVM. Regressao Logistica, KNN e SVM usam `StandardScaler`
dentro do pipeline para evitar vazamento de dados em validacoes futuras.
Arvore de Decisao e Random Forest nao usam padronizacao.

Nenhum modelo foi treinado ou selecionado ainda.

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
