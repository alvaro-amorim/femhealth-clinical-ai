# FemHealth ML Triage

Projeto academico para estruturar uma base inicial de triagem em saude feminina com
apoio de Machine Learning.

O objetivo geral e criar, em etapas futuras, um fluxo reprodutivel que va de um
notebook de treinamento para um modelo serializado em Joblib, consumido por uma API
FastAPI e apresentado em uma interface Streamlit.

Estado atual: benchmark e ajuste de hiperparâmetros concluídos, SVM calibrado
selecionado, avaliação final do holdout concluída, artefato final persistido e
API de inferência criada.

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
teste final. O conjunto de desenvolvimento foi usado para comparação baseline,
ajuste e seleção. O teste final foi reservado até a avaliação final congelada.

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

Naquela etapa, nenhum modelo final havia sido selecionado ou persistido, e o
conjunto de teste final permanecia intocado.

## Ajuste de hiperparâmetros

Regressão Logística, Random Forest e SVM foram ajustados com `GridSearchCV`,
validação cruzada estratificada de 5 folds e `recall_malignant` como métrica
principal de refit. As demais métricas do benchmark também foram avaliadas.

KNN e Árvore de Decisão permaneceram documentados no benchmark, mas não
foram ajustados naquela etapa. Durante o ajuste, o holdout ainda não havia sido
acessado, e nenhum modelo final havia sido selecionado ou persistido.

## Probabilidades e thresholds

Os melhores hiperparâmetros ajustados foram reutilizados para gerar
probabilidades out-of-fold somente no conjunto de desenvolvimento. Foram
comparadas probabilidades nativas e calibradas por sigmoid, com thresholds de
0.05 a 0.95.

A análise usou recall maligno mínimo de 0.97 como restrição acadêmica para
thresholds provisórios antes do congelamento da seleção. Esses pontos não eram
recomendações clínicas. Naquela etapa, o teste final ainda permanecia intocado.

## Seleção congelada e resultado final

SVM com kernel RBF, `StandardScaler` no pipeline, calibração sigmoid e threshold
0.51 foi selecionado antes da avaliação final, usando somente estimativas do
conjunto de desenvolvimento. A execução final foi realizada por:

```powershell
.\.venv\Scripts\python.exe -m femhealth.final_run
```

No holdout final de 114 registros, o resultado foi:

- accuracy: 0.9737;
- balanced accuracy: 0.9742;
- precision maligno: 0.9535;
- recall maligno: 0.9762;
- F1 maligno: 0.9647;
- specificity benigno: 0.9722;
- ROC AUC maligno: 0.9940;
- average precision maligno: 0.9918;
- Brier Score: 0.0278;
- log loss: 0.0950.

Matriz de confusão:

- malignos corretos: 41;
- falsos negativos malignos: 1;
- falsos positivos malignos: 2;
- benignos corretos: 70.

O desempenho final manteve o objetivo observado no desenvolvimento: recall
maligno alto com especificidade benigna também alta. Nenhuma decisão de modelo,
calibração ou threshold foi alterada após abrir o holdout. O projeto não possui
validade clínica.

## Artefato do modelo

O estimador final é treinado apenas nos 455 registros de desenvolvimento. Os
114 registros do holdout não entram no treinamento.

O artefato foi gerado por:

```powershell
.\.venv\Scripts\python.exe -m femhealth.model_artifact
```

Arquivos esperados:

- `artifacts/model/femhealth_svm_sigmoid.joblib`;
- `artifacts/model/femhealth_svm_sigmoid.metadata.json`.

Os metadados registram o contrato de 30 features na ordem canônica, threshold
0.51, classes, divisão dos dados, métricas finais já salvas e SHA-256 do Joblib.
O carregamento validado recalcula o hash antes de carregar o estimador.

O artefato foi gerado com Scikit-learn 1.9.0 e deve ser carregado com essa mesma versão.

Arquivos Joblib devem ser carregados somente de fonte confiável. API e
Streamlit deverão usar esse artefato sem retreinamento. O modelo permanece
acadêmico e não possui validade clínica.

## API de inferência

A API carrega o modelo uma vez durante o `lifespan`. Nenhuma requisição treina,
ajusta ou altera o modelo.

Endpoints:

- `GET /health`;
- `GET /model`;
- `POST /predict`.

Execução local:

```powershell
.\.venv\Scripts\python.exe -m uvicorn femhealth.api:app --host 127.0.0.1 --port 8000
```

A documentação interativa fica disponível em:

```text
http://127.0.0.1:8000/docs
```

O endpoint `POST /predict` exige um objeto `features` com as 30 chaves
canônicas. A ordem recebida é normalizada para a ordem canônica antes da
inferência. O resultado é acadêmico e não possui validade clínica.

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
