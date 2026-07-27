# Ética, limitações e uso responsável

## Natureza acadêmica

FemHealth Clinical AI é um projeto acadêmico. Ele demonstra um fluxo técnico de
classificação tabular, mas não deve ser interpretado como solução clínica.

## Limitações dos dados

O projeto usa o Breast Cancer Wisconsin Diagnostic (WDBC), com 569 registros e
30 variáveis numéricas. O dataset é específico, histórico e não contém o contexto
clínico completo de uma pessoa.

O dataset atual não representa toda a saúde feminina e não inclui imagens,
mamografias, histórico longitudinal ou informações clínicas amplas.

## Riscos de generalização

O desempenho observado no WDBC pode mudar em outros cenários, populações,
equipamentos, protocolos de coleta ou formas de medição. A ausência de validação
externa impede qualquer conclusão sobre generalização clínica.

## Falsos negativos e falsos positivos

Um falso negativo maligno ocorre quando um caso real maligno é classificado como
benigno pelo modelo. Um falso positivo maligno ocorre quando um caso real benigno
é classificado como maligno.

Esses erros têm implicações diferentes, mas este documento não fornece
orientação médica, conduta, tratamento ou priorização clínica.

## Interpretação das probabilidades

A probabilidade produzida pelo modelo não é uma probabilidade clínica real. Ela
é uma saída estatística estimada dentro do protocolo, dataset e calibração
usados no projeto.

A calibração é limitada ao WDBC e ao protocolo executado. O limiar de decisão
0.51 é uma escolha acadêmica congelada para este experimento e não deve ser
transferido para prática clínica.

## Casos de demonstração conhecidos

Oito registros do holdout final foram materializados após a avaliação final
congelada para demonstração acadêmica. A regra foi usar os primeiros oito
registros na ordem congelada do holdout final, sem selecionar por acerto, erro,
probabilidade, confiança, dificuldade ou classe prevista.

Esses registros não foram usados no treinamento nem na seleção atual do modelo,
mas passam a ser exemplos públicos conhecidos a partir desta versão. Portanto,
futuras versões não devem usá-los para seleção de modelo ou avaliação
independente.

A taxa exibida pela página de demonstração descreve somente os casos únicos
executados na sessão atual. Ela não substitui a acurácia oficial do holdout
completo de 114 registros.

Uma futura avaliação exige novos dados externos ou novo conjunto preservado.

## Explicabilidade responsável

A importância por permutação mede dependência preditiva global. Ela não mede
causalidade, não comprova relevância clínica e pode ser afetada por correlação
entre variáveis.

Diferenças pequenas no ranking devem ser interpretadas com cautela. O projeto
não oferece explicação clínica individual.

## Supervisão humana

Qualquer estudo futuro relacionado a saúde deve envolver profissionais
qualificados, revisão metodológica, governança e avaliação ética apropriada.

## Privacidade e segurança

O repositório usa dados públicos do WDBC e não implementa banco de dados de
pacientes. O código atual não comprova conformidade com LGPD, HIPAA ou outras
normas regulatórias.

Arquivos Joblib devem ser carregados somente de fontes confiáveis, pois esse
formato executa desserialização de objetos Python.

## Usos proibidos

Usos proibidos:

- diagnóstico;
- triagem clínica real;
- decisão médica autônoma;
- indicação de tratamento;
- uso emergencial;
- substituição de avaliação profissional;
- uso em produção médica.

## Requisitos antes de qualquer estudo clínico futuro

Antes de qualquer estudo clínico futuro seriam necessários:

- aprovação ética;
- governança de dados;
- validação externa;
- avaliação prospectiva;
- análise de viés;
- avaliação de segurança;
- participação de profissionais qualificados.

## Declaração final

Este projeto deve ser lido como exercício acadêmico de engenharia de dados,
modelagem, inferência e documentação responsável. Ele não estabelece validade
clínica.
