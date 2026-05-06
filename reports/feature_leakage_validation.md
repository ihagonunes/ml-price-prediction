# Validacao do Pipeline de Features (Leakage e Consistencia)

- Gerado em: `2026-05-05T21:13:27.522879`
- Fonte de features: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\features_temporal`
- Fonte dos splits: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical_curated`

## Resultado DE

- Folds avaliados: `20` (`0` com falha).
- Overlap total de RideID entre treino e validacao: `0`.
- Overlap total de indices entre treino e validacao: `0`.
- Regra temporal validada em todos os folds: `max(train_date) < min(validation_date)`.

## Resultado DS

- Features historicas por usuario foram revisadas como sem leakage quando usadas com Create como ancora temporal.
- Features com risco alto foram identificadas e marcadas para exclusao do baseline de modelagem.

## Checagens de Consistencia das Features Historicas

- Inconsistencias de UserPriorRideCount por RideID: `0`.
- Inconsistencias de UserPriorCategoryRideCount por RideID+CategoryID: `0`.
- Mismatch contra cumcount esperado (ride): `0`.
- Mismatch contra cumcount esperado (categoria): `0`.
- Linhas com UserPriorCategoryRideCount > 0 e media nula: `0`.
- Linhas com UserPriorRideCount > 0 e UserPriorPaidPriceMean nula: `426` (esperado quando nao existe RidePrice historico valido > 0).

## Tabela de Risco de Leakage por Grupo de Feature

| feature_group                                           | leakage_risk   | recommended_decision                                              | rationale                                                                                                                  |
|:--------------------------------------------------------|:---------------|:------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| Create* temporais                                       | low            | Manter no baseline.                                               | Derivadas do timestamp de criacao da corrida, disponivel no momento de inferencia.                                         |
| Schedule* temporais                                     | low            | Manter, desde que Schedule esteja disponivel na predicao.         | Timestamp de agendamento e pre-evento; nao depende de resultado futuro da corrida.                                         |
| Updated* temporais                                      | high           | Excluir dos modelos de preco.                                     | Updated e posterior a Create em 100.0% das linhas, indicando forte risco de carregar informacao pos-evento.                |
| UserPriorRideCount / UserPriorPaidPriceMean             | low            | Manter no baseline.                                               | Calculadas com janela expansiva por UserID no nivel de RideID, sempre excluindo a corrida atual.                           |
| UserPriorCategoryRideCount / UserPriorCategoryPriceMean | low            | Manter no baseline.                                               | Calculadas com janela expansiva por UserID + CategoryID no nivel canonico RideID+CategoryID, sem uso de informacao futura. |
| PriceWasCapped                                          | high           | Excluir dos modelos de preco.                                     | Indicador derivado do proprio target Price no preprocessamento (ativo em 0.4991% das linhas).                              |
| WaitingTimeWasCapped / FareIDWasImputed                 | low            | Uso opcional; nao sao necessarias no baseline.                    | Flags de qualidade/transformacao de entrada; nao usam informacao futura do target.                                         |
| Cross own price (Price_UberX)                           | high           | Excluir da matriz de features do proprio modelo.                  | Para linhas de UberX, Price_UberX coincide com o target Price em 99.9788% das linhas, caracterizando leakage direto.       |
| Cross own price (Price_Comfort)                         | high           | Excluir da matriz de features do proprio modelo.                  | Para linhas de Comfort, Price_Comfort coincide com o target Price em 99.9777% das linhas, caracterizando leakage direto.   |
| Cross own price (Price_Black)                           | high           | Excluir da matriz de features do proprio modelo.                  | Para linhas de Black, Price_Black coincide com o target Price em 99.9741% das linhas, caracterizando leakage direto.       |
| Cross auxiliar (UberX)                                  | medium         | Manter com controle de disponibilidade e sem usar coluna propria. | As duas features auxiliares estao disponiveis em 50.2376% das linhas de UberX.                                             |
| Cross auxiliar (Comfort)                                | medium         | Manter com controle de disponibilidade e sem usar coluna propria. | As duas features auxiliares estao disponiveis em 61.3622% das linhas de Comfort.                                           |
| Cross auxiliar (Black)                                  | medium         | Manter com controle de disponibilidade e sem usar coluna propria. | As duas features auxiliares estao disponiveis em 95.7478% das linhas de Black.                                             |

## Features de Alto Risco (Acao Obrigatoria)

| feature_group                   | leakage_risk   | recommended_decision                             | rationale                                                                                                                |
|:--------------------------------|:---------------|:-------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| Updated* temporais              | high           | Excluir dos modelos de preco.                    | Updated e posterior a Create em 100.0% das linhas, indicando forte risco de carregar informacao pos-evento.              |
| PriceWasCapped                  | high           | Excluir dos modelos de preco.                    | Indicador derivado do proprio target Price no preprocessamento (ativo em 0.4991% das linhas).                            |
| Cross own price (Price_UberX)   | high           | Excluir da matriz de features do proprio modelo. | Para linhas de UberX, Price_UberX coincide com o target Price em 99.9788% das linhas, caracterizando leakage direto.     |
| Cross own price (Price_Comfort) | high           | Excluir da matriz de features do proprio modelo. | Para linhas de Comfort, Price_Comfort coincide com o target Price em 99.9777% das linhas, caracterizando leakage direto. |
| Cross own price (Price_Black)   | high           | Excluir da matriz de features do proprio modelo. | Para linhas de Black, Price_Black coincide com o target Price em 99.9741% das linhas, caracterizando leakage direto.     |

## Evidencia dos Folds

| scope    | split_name   | split_type   |   category_id | category_name   |   train_rows |   evaluation_rows | train_start   | train_end   | gap_start   | gap_end    | evaluation_start   | evaluation_end   |   index_overlap |   ride_overlap | temporal_order_ok   | timestamp_order_ok   | gap_order_ok   | fold_valid   |
|:---------|:-------------|:-------------|--------------:|:----------------|-------------:|------------------:|:--------------|:------------|:------------|:-----------|:-------------------|:-----------------|----------------:|---------------:|:--------------------|:---------------------|:---------------|:-------------|
| category | fold_1       | cv           |             2 | UberX           |        27535 |              7647 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_2       | cv           |             2 | UberX           |        33773 |              8940 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_3       | cv           |             2 | UberX           |        41871 |             12896 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_4       | cv           |             2 | UberX           |        54095 |             15752 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |               0 |              0 | True                | True                 | True           | True         |
| category | holdout_test | holdout      |             2 | UberX           |        73490 |             16456 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_1       | cv           |             4 | Uber Black      |         8052 |              2604 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_2       | cv           |             4 | Uber Black      |        10093 |              3342 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_3       | cv           |             4 | Uber Black      |        12992 |              5873 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_4       | cv           |             4 | Uber Black      |        18541 |              6238 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |               0 |              0 | True                | True                 | True           | True         |
| category | holdout_test | holdout      |             4 | Uber Black      |        26385 |              7004 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_1       | cv           |             9 | Uber Comfort    |        14563 |              5536 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_2       | cv           |             9 | Uber Comfort    |        18980 |              6580 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_3       | cv           |             9 | Uber Comfort    |        24772 |             10979 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |               0 |              0 | True                | True                 | True           | True         |
| category | fold_4       | cv           |             9 | Uber Comfort    |        35107 |             11063 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |               0 |              0 | True                | True                 | True           | True         |
| category | holdout_test | holdout      |             9 | Uber Comfort    |        49289 |             12038 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |               0 |              0 | True                | True                 | True           | True         |
| overall  | fold_1       | cv           |           nan | all             |       104318 |             38660 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |               0 |              0 | True                | True                 | True           | True         |
| overall  | fold_2       | cv           |           nan | all             |       135617 |             44373 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |               0 |              0 | True                | True                 | True           | True         |
| overall  | fold_3       | cv           |           nan | all             |       175178 |             72428 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |               0 |              0 | True                | True                 | True           | True         |
| overall  | fold_4       | cv           |           nan | all             |       242864 |             83632 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |               0 |              0 | True                | True                 | True           | True         |
| overall  | holdout_test | holdout      |           nan | all             |       347134 |             90668 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |               0 |              0 | True                | True                 | True           | True         |