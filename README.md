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

O WDBC é separado de forma estratificada em 80% para desenvolvimento e 20% para
teste final. O conjunto de desenvolvimento é usado para comparação baseline. O
teste final permanece intocado para avaliação posterior.

## Modelos candidatos

Os candidatos planejados são Regressão Logística, KNN, Árvore de Decisão,
Random Forest e SVM. Regressão Logística, KNN e SVM usam `StandardScaler`
dentro do pipeline para evitar vazamento de dados em validações futuras.
Árvore de Decisão e Random Forest não usam padronização.

## Benchmark baseline

O benchmark baseline já foi executado com validação cruzada estratificada de 5
folds usando somente o conjunto de desenvolvimento. Os cinco modelos foram
treinados temporariamente dentro dos folds para estimar métricas de comparação:
accuracy, balanced accuracy, precision, recall e F1 para a classe maligna,
specificity para benignos, ROC AUC e average precision.

Nenhum modelo final foi selecionado ou persistido. O conjunto de teste final
continua intocado.

## Ajuste de hiperparâmetros

Regressão Logística, Random Forest e SVM foram ajustados com `GridSearchCV`,
validação cruzada estratificada de 5 folds e `recall_malignant` como métrica
principal de refit. As demais métricas do benchmark também são avaliadas.

KNN e Árvore de Decisão permanecem documentados no benchmark, mas não são
ajustados nesta etapa. O teste final segue intocado, e nenhum modelo final foi
selecionado ou persistido.

## Probabilidades e thresholds

Os melhores hiperparâmetros ajustados foram reutilizados para gerar
probabilidades out-of-fold somente no conjunto de desenvolvimento. Foram
comparadas probabilidades nativas e calibradas por sigmoid, com thresholds de
0.05 a 0.95.

A análise usa recall maligno mínimo de 0.97 como restrição acadêmica para
thresholds provisórios. Esses pontos não são recomendações clínicas. O teste
final permanece intocado, e nenhum modelo final foi selecionado ou persistido.

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
