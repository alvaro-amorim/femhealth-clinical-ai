# Documentação acadêmica — FemHealth Clinical AI

## 1. Resumo do projeto

FemHealth Clinical AI é um projeto acadêmico de classificação de padrões do
Breast Cancer Wisconsin Diagnostic (WDBC). O objetivo é demonstrar, de forma
reprodutível, um fluxo técnico de aprendizado de máquina com validação de dados,
comparação de modelos, calibração, avaliação final, persistência de artefato,
API, interface e explicabilidade global.

O projeto tem finalidade educacional e técnica. Ele não possui validade clínica,
não deve ser usado para decisão médica e não representa toda a área de saúde
feminina.

## 2. Problema abordado

O problema tratado é uma classificação binária de padrões tabulares do WDBC. A
classe `0` representa `malignant` e a classe `1` representa `benign`.

Como a classe de interesse é a classe `0`, o projeto localiza explicitamente a
probabilidade maligna na coluna associada a essa classe, em vez de assumir a
orientação padrão de bibliotecas de modelagem.

O sistema não interpreta imagens, mamografias ou exames clínicos completos. O
dataset atual é tabular e contém medidas numéricas já estruturadas.

## 3. Dados

O dataset utilizado é o Breast Cancer Wisconsin Diagnostic, carregado
exclusivamente por `sklearn.datasets.load_breast_cancer(as_frame=True)`.

Características comprovadas pelo contrato do projeto:

| Item | Valor |
| --- | ---: |
| Registros | 569 |
| Variáveis numéricas | 30 |
| Casos malignos | 212 |
| Casos benignos | 357 |
| Valores ausentes | 0 |

As 30 variáveis possuem ordem canônica preservada pelo contrato automatizado.
Essa ordem é usada no treinamento, na inferência e na validação da API.

A exploração inicial está documentada em
[`notebooks/01_exploracao_wdbc.ipynb`](../notebooks/01_exploracao_wdbc.ipynb).

## 4. Protocolo experimental

O protocolo foi construído em etapas:

1. contrato e exploração inicial do WDBC;
2. separação estratificada entre desenvolvimento e teste final;
3. benchmark baseline com cinco modelos;
4. ajuste de hiperparâmetros dos três modelos mais promissores;
5. calibração de probabilidades;
6. análise de limiar de decisão no desenvolvimento;
7. congelamento da seleção final antes de abrir o holdout;
8. avaliação única do holdout final;
9. persistência do estimador selecionado;
10. explicabilidade global por importância de permutação.

A separação usou `random_state=42` e manteve o teste final fora das decisões de
seleção, calibração e limiar.

| Conjunto | Registros | Malignos | Benignos |
| --- | ---: | ---: | ---: |
| Desenvolvimento | 455 | 170 | 285 |
| Teste final | 114 | 42 | 72 |

## 5. Modelos avaliados

| Modelo | Comparado no benchmark | Ajustado com hiperparâmetros |
| --- | --- | --- |
| Regressão Logística | Sim | Sim |
| KNN | Sim | Não |
| Árvore de Decisão | Sim | Não |
| Random Forest | Sim | Sim |
| SVM | Sim | Sim |

Essa tabela documenta o processo seguido. Ela não representa uma nova seleção de
modelo.

## 6. Modelo selecionado

A seleção final congelada foi a variante `svm_sigmoid`:

| Item | Valor |
| --- | --- |
| Modelo base | SVM |
| Kernel | RBF |
| Padronização | `StandardScaler` dentro do `Pipeline` |
| Calibração | sigmoid |
| Limiar maligno | 0.51 |
| Classe maligna | 0 |
| Classe benigna | 1 |

A decisão foi tomada antes da avaliação final do holdout, usando somente
estimativas do conjunto de desenvolvimento.

## 7. Resultado final

O holdout final foi avaliado uma única vez, após o congelamento da seleção.

| Métrica | Valor |
| --- | ---: |
| Accuracy | 0.9736842105263158 |
| Balanced accuracy | 0.9742063492063492 |
| Precision maligno | 0.9534883720930233 |
| Recall maligno | 0.9761904761904762 |
| F1 maligno | 0.9647058823529412 |
| Especificidade benigna | 0.9722222222222222 |
| ROC AUC maligno | 0.9940476190476191 |
| Average precision maligno | 0.9917642411473949 |
| Brier Score | 0.0277539756400032 |
| Log loss | 0.09504263948901345 |

Matriz de confusão:

| Real / Previsto | Maligno | Benigno |
| --- | ---: | ---: |
| Maligno | 41 | 1 falso negativo |
| Benigno | 2 falsos positivos | 70 |

A interpretação deve ser cautelosa. O teste final contém apenas 42 casos
malignos; por isso, cada erro tem impacto relevante nas métricas. Também há
apenas uma divisão de teste e não existe validação externa independente.

### Casos de demonstração materializados

Após a avaliação final já congelada, oito registros do holdout final foram
materializados em `artifacts/demo/holdout_demo_cases.json` para demonstração
acadêmica em vídeo. A regra de seleção foi: primeiros oito registros na ordem
congelada do holdout final.

Os índices são `256`, `428`, `501`, `363`, `564`, `464`, `358` e `343`, com
quatro registros `malignant` e quatro registros `benign`. Eles não foram
selecionados por acerto, erro, probabilidade, confiança, dificuldade ou classe
prevista; uma divergência está presente naturalmente, mas não foi usada como
critério.

Esses oito registros nunca foram usados no treinamento nem na seleção atual do
modelo. A página de demonstração executa cada caso pelo endpoint `POST /predict`
existente e compara o rótulo de referência do dataset com a classificação
retornada. A taxa exibida na sessão é apenas descritiva dos casos únicos
executados naquela sessão e não substitui a acurácia oficial dos 114 registros.

Como esses oito registros passam a ser exemplos públicos conhecidos, futuras
versões do modelo não devem usá-los para seleção ou avaliação independente. Uma
futura avaliação exige novos dados externos ou novo conjunto preservado.

## 8. Explicabilidade

A explicabilidade global foi calculada por importância de permutação, usando
somente os 455 registros de desenvolvimento em validação cruzada estratificada
com 5 folds. Foram realizadas 10 repetições por variável e fold, totalizando
1.500 observações detalhadas.

Os artefatos persistidos indicam `holdout_used=false`. A métrica usada foi ROC
AUC da classe maligna, com ROC AUC médio entre folds aproximadamente 0.996285.
A maior importância média registrada foi `worst texture`.

A importância por permutação mede dependência preditiva, não causalidade.
Variáveis correlacionadas podem compartilhar importância, e nenhuma variável foi
selecionada ou removida com base nesse ranking.

## 9. Arquitetura da aplicação

O fluxo geral é:

dados e treinamento → artefatos congelados → FastAPI → Streamlit.

A FastAPI carrega os artefatos no `lifespan`; o Streamlit consome a API por
HTTP. A arquitetura completa está descrita em
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## 10. Reprodutibilidade

O projeto usa Python 3.11. Para desenvolvimento completo, incluindo testes e
componentes de análise:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Os notebooks executados são:

- [`notebooks/01_exploracao_wdbc.ipynb`](../notebooks/01_exploracao_wdbc.ipynb);
- [`notebooks/02_modelagem_e_explicabilidade.ipynb`](../notebooks/02_modelagem_e_explicabilidade.ipynb).

O artefato final está em
[`artifacts/model/femhealth_svm_sigmoid.joblib`](../artifacts/model/femhealth_svm_sigmoid.joblib)
e possui SHA-256:

```text
CC43CEC3BA58C5A4950217E80C8B286B0E7DB501FF663BA9BE9F91DF1F4B05B5
```

## 11. Limitações

As principais limitações são:

- dataset pequeno e histórico;
- ausência de validação externa;
- ausência de contexto clínico completo;
- uso de dados tabulares, sem imagens;
- possível correlação entre variáveis;
- incerteza na estabilidade das importâncias;
- risco de falsos negativos e falsos positivos;
- impossibilidade de generalizar o resultado para saúde feminina em geral.

## 12. Conclusão

O projeto consolidou um fluxo acadêmico completo para classificação tabular do
WDBC, com contrato de dados, benchmark, tuning, calibração, avaliação final
congelada, artefato versionado, API, interface e explicabilidade global.

Os resultados são tecnicamente consistentes com o protocolo executado, mas não
estabelecem validade clínica nem autorização de uso em contexto real.

## 13. Próximos passos

Possíveis próximos passos acadêmicos:

- elaboração do relatório final em PDF;
- gravação de vídeo demonstrativo;
- validação externa como trabalho futuro;
- estudo de dados adicionais;
- visão computacional como extensão futura independente.

Esses itens não estão implementados nesta versão.
