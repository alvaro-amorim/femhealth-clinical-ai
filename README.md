# FemHealth Clinical AI

Projeto acadêmico de classificação de padrões do Breast Cancer Wisconsin
Diagnostic (WDBC), desenvolvido com apoio de aprendizado de máquina.

O repositório implementa um fluxo reproduzível que parte da validação e análise
dos dados, passa por comparação, ajuste, calibração, avaliação e
explicabilidade, persiste o modelo em Joblib e disponibiliza inferência por
FastAPI e Streamlit.

Estado atual: benchmark e ajuste de hiperparâmetros concluídos, SVM calibrado
selecionado, avaliação final do holdout concluída, artefato final persistido,
FastAPI e interface Streamlit concluídas, explicabilidade global registrada e
integrada à aplicação, e notebook técnico consolidado.

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

Arquivos Joblib devem ser carregados somente de fonte confiável. A FastAPI usa
esse artefato sem retreinamento, e o Streamlit consome a inferência
exclusivamente pela API. O modelo permanece acadêmico e não possui validade
clínica.

## API de inferência

A API carrega o modelo uma vez durante o `lifespan`. Nenhuma requisição treina,
ajusta ou altera o modelo.

Endpoints:

- `GET /health`;
- `GET /model`;
- `GET /explainability`;
- `GET /explainability/plot`;
- `POST /predict`.

Os artefatos de explicabilidade são validados e carregados uma vez no
`lifespan`. Os endpoints apenas apresentam resultados persistidos; nenhuma
importância é recalculada por requisição.

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

## Interface Streamlit

A interface Streamlit é em português, usa tema claro e consome exclusivamente a
FastAPI por HTTP. O Streamlit não carrega o Joblib e toda inferência passa pela
API.

Páginas:

- Apresentação;
- Modelo e resultados;
- Explicabilidade;
- Simulador.

A página de explicabilidade consome JSON e PNG pela FastAPI. O Streamlit não lê
diretamente os arquivos em `reports/explainability`.

A API deve ser iniciada primeiro.

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn femhealth.api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Endereço padrão:

```text
http://localhost:8501
```

A variável `FEMHEALTH_API_URL` define a URL da API usada pela interface. O padrão
é:

```text
http://127.0.0.1:8000
```

O simulador exige um objeto `features` com as 30 chaves canônicas. A ordem
recebida é normalizada antes do envio à API. O resultado é acadêmico e não
possui validade clínica.

## Explicabilidade global

O SVM RBF não possui uma importância nativa simples por coeficientes. Por isso,
a explicabilidade global usa importância por permutação em validação cruzada
estratificada de 5 folds.

A análise usa somente os 455 registros de desenvolvimento. O holdout não é
reutilizado. A métrica analisada é ROC AUC da classe maligna.

Comando:

```powershell
.\.venv\Scripts\python.exe -m femhealth.explainability_run
```

Arquivos gerados:

- `reports/explainability/permutation_importance_details.csv`;
- `reports/explainability/permutation_importance_summary.csv`;
- `reports/explainability/permutation_importance_fold_scores.csv`;
- `reports/explainability/permutation_importance_metadata.json`;
- `reports/explainability/permutation_importance_top15.png`.

Resumo observado:

| Feature | Importância média | Desvio-padrão |
| --- | ---: | ---: |
| `worst texture` | 0.005593 | 0.004609 |
| `radius error` | 0.002023 | 0.002566 |
| `worst radius` | 0.001992 | 0.003237 |
| `worst area` | 0.001775 | 0.003215 |
| `worst concave points` | 0.001672 | 0.002882 |

As diferenças entre várias posições do ranking são pequenas em relação à
variabilidade observada. Por isso, o ranking deve ser interpretado como uma
indicação global aproximada de dependência preditiva, e não como uma ordem
precisa de relevância clínica.

Importância por permutação não representa causalidade. Variáveis correlacionadas
podem compartilhar importância. O ranking não removeu variáveis, não alterou o
modelo e não mudou o threshold. A análise não possui validade clínica.

## Notebook técnico

O notebook técnico executado está em
`notebooks/02_modelagem_e_explicabilidade.ipynb`.

Ele consolida benchmark, tuning, calibração, seleção congelada, resultado final,
artefato persistido e explicabilidade global. O holdout final não é reavaliado
pelo notebook, e a explicabilidade é lida dos artefatos já salvos. O notebook de
EDA permanece separado em `notebooks/01_exploracao_wdbc.ipynb`.

Para executar novamente em desenvolvimento:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert `
  --to notebook `
  --execute notebooks/02_modelagem_e_explicabilidade.ipynb `
  --inplace `
  --ExecutePreprocessor.timeout=900
```

Essa execução refaz somente as análises de desenvolvimento e lê os resultados
finais já persistidos.

Este projeto requer Python 3.11.

## Instalar

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
```

O extra `analysis` é recomendado para o desenvolvimento completo, pois os
testes atuais também importam módulos de análise e explicabilidade.

## Validar

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Este projeto é acadêmico e não é uma ferramenta clínica validada.
