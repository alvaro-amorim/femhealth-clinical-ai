# Model Card — FemHealth SVM Sigmoid

## Identificação

| Campo | Valor |
| --- | --- |
| Projeto | FemHealth Clinical AI |
| Variante | `svm_sigmoid` |
| Versão do artefato | 1.0.0 |
| Caminho | `artifacts/model/femhealth_svm_sigmoid.joblib` |
| SHA-256 | `CC43CEC3BA58C5A4950217E80C8B286B0E7DB501FF663BA9BE9F91DF1F4B05B5` |
| Python | 3.11 |
| Scikit-learn | 1.9.0 |

## Visão geral

O modelo classifica padrões tabulares do Breast Cancer Wisconsin Diagnostic
(WDBC) em duas classes: `0 = malignant` e `1 = benign`.

Ele não interpreta imagens, não lê mamografias, não usa prontuários e não emite
diagnóstico médico. A saída é uma classificação acadêmica baseada em 30
variáveis numéricas do WDBC.

## Uso pretendido

Usos pretendidos:

- estudo acadêmico;
- demonstração técnica;
- experimentação educacional.

## Usos não pretendidos

O artefato não deve ser usado para:

- diagnóstico;
- triagem clínica real;
- indicação de tratamento;
- substituição de profissional de saúde;
- decisão autônoma;
- uso emergencial;
- uso em produção médica.

## Dados

O dataset é o Breast Cancer Wisconsin Diagnostic, carregado por
`sklearn.datasets.load_breast_cancer(as_frame=True)`.

| Item | Valor |
| --- | ---: |
| Registros totais | 569 |
| Variáveis numéricas | 30 |
| Casos malignos | 212 |
| Casos benignos | 357 |
| Desenvolvimento | 455 |
| Desenvolvimento maligno | 170 |
| Desenvolvimento benigno | 285 |
| Teste final | 114 |
| Teste maligno | 42 |
| Teste benigno | 72 |

A separação foi estratificada com `random_state=42`. O holdout ficou fora do
treinamento do artefato final.

## Variáveis de entrada

O modelo exige exatamente as 30 variáveis canônicas do WDBC, na ordem definida
pelo contrato do projeto. A exploração detalhada das variáveis está em
[`notebooks/01_exploracao_wdbc.ipynb`](../notebooks/01_exploracao_wdbc.ipynb).

## Pré-processamento e modelo

O pré-processamento usa `StandardScaler` dentro do `Pipeline`, evitando aplicar
padronização antes da validação cruzada ou fora do estimador.

Configuração congelada:

- SVM com kernel RBF;
- calibração sigmoid;
- `C=1.0`;
- `gamma="scale"`;
- `class_weight="balanced"`;
- limiar de decisão maligno 0.51;
- classe maligna 0;
- probabilidade maligna localizada explicitamente na classe 0.

## Processo de seleção

O processo incluiu benchmark com Regressão Logística, KNN, Árvore de Decisão,
Random Forest e SVM. Depois, Regressão Logística, Random Forest e SVM foram
ajustados com validação cruzada estratificada.

As probabilidades out-of-fold foram comparadas em variantes nativas e
calibradas. A seleção `svm_sigmoid` e o limiar 0.51 foram congelados antes da
avaliação final do holdout.

## Métricas

Resultado no holdout final de 114 registros:

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

## Explicabilidade

A explicabilidade global usa importância por permutação em validação cruzada
estratificada de 5 folds, somente no conjunto de desenvolvimento. Foram 10
repetições por variável e fold, com 1.500 observações detalhadas.

O ROC AUC médio registrado entre folds é aproximadamente 0.996285. A maior
importância média registrada foi `worst texture`.

Essa análise não indica causalidade, não define relevância clínica e não remove
variáveis do modelo.

## Limitações e riscos

Limitações e riscos relevantes:

- tamanho reduzido do dataset;
- baixa diversidade conhecida;
- origem histórica e específica do WDBC;
- ausência de validação externa;
- correlação entre variáveis;
- possibilidade de falsos negativos;
- possibilidade de falsos positivos;
- dependência de medidas corretamente obtidas;
- ausência de contexto clínico;
- possível mudança de desempenho em outras populações;
- não generalização para saúde feminina em geral.

## Considerações éticas

As restrições de uso e interpretação estão detalhadas em
[`docs/RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## Manutenção e governança

O artefato atual é congelado. Qualquer alteração de modelo, limiar de decisão,
features, calibração ou dados deve iniciar um novo ciclo metodológico.

O teste final atual não deve ser usado repetidamente para tomar novas decisões.
Uma nova versão exige novo artefato, novos metadados, documentação atualizada e
validação correspondente.
