# Analise de Correlacao com o Target Price

- Gerado em: `2026-05-04T22:04:05.623439`
- Fonte: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical`
- Carregamento em memoria: `2000000` linhas x `15` colunas (`~114.44 MB` em `float32`).
- Leitura feita diretamente do Parquet consolidado, selecionando apenas colunas numericas ou numerico-textuais convertiveis.

## Variaveis Incluidas

| feature                        | dtype   |   null_pct |   nunique | feature_family     | feature_status        |
|:-------------------------------|:--------|-----------:|----------:|:-------------------|:----------------------|
| WaitingTime                    | int64   |       0    |        62 | continuous         | candidate             |
| Price                          | float64 |       0    |     17239 | target             | candidate             |
| Selected                       | int64   |       0    |         2 | post_event         | leakage_or_post_event |
| RideReasonSelectedEstimativeID | float64 |      88.3  |         2 | post_event         | leakage_or_post_event |
| Fee                            | float64 |       0    |         4 | monetary_component | candidate             |
| RideStatusID                   | int64   |       0    |         3 | categorical_code   | candidate             |
| CompanyID                      | int64   |       0    |        46 | categorical_code   | candidate             |
| RidePrice                      | float64 |       0    |      9478 | leakage            | leakage_or_post_event |
| TotalUsers                     | int64   |       0    |         5 | count              | candidate             |
| ProductProviderID              | int64   |       0    |         4 | categorical_code   | candidate             |
| CategoryID                     | Int64   |       0    |         8 | categorical_code   | candidate             |
| OriginLat                      | Float64 |       0    |     45951 | spatial_coordinate | candidate             |
| OriginLng                      | Float64 |       0    |     46045 | spatial_coordinate | candidate             |
| DestinationLat                 | Float64 |       0.01 |     21745 | spatial_coordinate | candidate             |
| DestinationLng                 | Float64 |       0.01 |     21697 | spatial_coordinate | candidate             |

## Variaveis Excluidas da Matriz

| feature                  |   null_pct |   nunique | exclusion_reason                                                   |
|:-------------------------|-----------:|----------:|:-------------------------------------------------------------------|
| RideEstimativeID         |       0    |   2000000 | Identificador unico da estimativa.                                 |
| RideID                   |       0    |    239270 | Identificador da corrida, nao interpretavel como feature numerica. |
| ProviderID               |      99.2  |         3 | Mais de 95% de nulos no dataset consolidado.                       |
| RideProviderID           |      99.32 |      1145 | Mais de 95% de nulos e alta cardinalidade residual.                |
| RideCategoryID           |      99.2  |         3 | Mais de 95% de nulos no dataset consolidado.                       |
| RideDriverLocationID     |      99.54 |       758 | Mais de 95% de nulos e sem interpretacao direta.                   |
| ScheduledRide            |       0    |         1 | Coluna constante no dataset analitico atual.                       |
| OriginRideAddressID      |       0    |    239270 | Identificador do endereco de origem.                               |
| DestinationRideAddressID |       0    |    239270 | Identificador do endereco de destino.                              |

## Correlacao com o Target

| feature                        |   corr_to_target |   abs_corr_to_target |   null_pct | feature_family     | modeling_role   | notes                                                                          |
|:-------------------------------|-----------------:|---------------------:|-----------:|:-------------------|:----------------|:-------------------------------------------------------------------------------|
| RidePrice                      |       0.833082   |           0.833082   |       0    | leakage            | exclude         | Preco real da corrida apos a execucao; vazamento explicito.                    |
| ProductProviderID              |       0.0572855  |           0.0572855  |       0    | categorical_code   | candidate       |                                                                                |
| Selected                       |      -0.0357551  |           0.0357551  |       0    | post_event         | exclude         | Marcador de escolha da estimativa, conhecido apenas apos a selecao do usuario. |
| RideStatusID                   |       0.0310755  |           0.0310755  |       0    | categorical_code   | candidate       |                                                                                |
| WaitingTime                    |       0.0295887  |           0.0295887  |       0    | continuous         | candidate       |                                                                                |
| DestinationLat                 |      -0.0193468  |           0.0193468  |       0.01 | spatial_coordinate | candidate       |                                                                                |
| OriginLat                      |      -0.0190202  |           0.0190202  |       0    | spatial_coordinate | candidate       |                                                                                |
| RideReasonSelectedEstimativeID |       0.015441   |           0.015441   |      88.3  | post_event         | exclude         | Motivo da selecao da estimativa, disponivel apenas apos o evento.              |
| TotalUsers                     |       0.00853481 |           0.00853481 |       0    | count              | candidate       |                                                                                |
| CompanyID                      |       0.00707074 |           0.00707074 |       0    | categorical_code   | candidate       |                                                                                |
| CategoryID                     |       0.00588312 |           0.00588312 |       0    | categorical_code   | candidate       |                                                                                |
| OriginLng                      |       0.00390427 |           0.00390427 |       0    | spatial_coordinate | candidate       |                                                                                |
| DestinationLng                 |       0.00215931 |           0.00215931 |       0.01 | spatial_coordinate | candidate       |                                                                                |
| Fee                            |      -0.00189278 |           0.00189278 |       0    | monetary_component | candidate       |                                                                                |

## Multicolinearidade

| feature_left   | feature_right   |     corr |   abs_corr |
|:---------------|:----------------|---------:|-----------:|
| OriginLng      | DestinationLng  | 0.999211 |   0.999211 |
| OriginLat      | DestinationLat  | 0.9992   |   0.9992   |

## Baixo Sinal Linear

| feature        |   corr_to_target |   abs_corr_to_target |   null_pct | feature_family     | possible_action                                |
|:---------------|-----------------:|---------------------:|-----------:|:-------------------|:-----------------------------------------------|
| RideStatusID   |       0.0310755  |           0.0310755  |       0    | categorical_code   | keep_as_categorical_test_not_by_pearson_alone  |
| WaitingTime    |       0.0295887  |           0.0295887  |       0    | continuous         | keep_for_baseline_due_business_plausibility    |
| DestinationLat |      -0.0193468  |           0.0193468  |       0.01 | spatial_coordinate | transform_into_distance_or_zone_features       |
| OriginLat      |      -0.0190202  |           0.0190202  |       0    | spatial_coordinate | transform_into_distance_or_zone_features       |
| TotalUsers     |       0.00853481 |           0.00853481 |       0    | count              | keep_for_baseline_due_business_plausibility    |
| CompanyID      |       0.00707074 |           0.00707074 |       0    | categorical_code   | keep_as_categorical_test_not_by_pearson_alone  |
| CategoryID     |       0.00588312 |           0.00588312 |       0    | categorical_code   | keep_as_categorical_test_not_by_pearson_alone  |
| OriginLng      |       0.00390427 |           0.00390427 |       0    | spatial_coordinate | transform_into_distance_or_zone_features       |
| DestinationLng |       0.00215931 |           0.00215931 |       0.01 | spatial_coordinate | transform_into_distance_or_zone_features       |
| Fee            |      -0.00189278 |           0.00189278 |       0    | monetary_component | deprioritize_raw_feature_or_validate_semantics |

## Interpretacao DS

- Sinal linear mais alto entre variaveis elegiveis: ProductProviderID (0.0573); RideStatusID (0.0311); WaitingTime (0.0296); DestinationLat (-0.0193); OriginLat (-0.0190).
- RidePrice tem correlacao 0.8331 com o target, mas deve ser excluida por vazamento.
- Multicolinearidade relevante detectada principalmente nas coordenadas de origem/destino: OriginLng x DestinationLng (0.9992); OriginLat x DestinationLat (0.9992).
- Baixo sinal linear global com o target nas colunas cruas: RideStatusID, WaitingTime, DestinationLat, OriginLat, TotalUsers, CompanyID, CategoryID, OriginLng, DestinationLng, Fee. Isso nao implica exclusao automatica para codigos categoricos ou variaveis espaciais transformaveis.
- Para feature engineering, a prioridade deve ser transformar variaveis espaciais em distancia/zona, tratar ProductProviderID e CategoryID como categoricas e derivar features temporais a partir de Create/Schedule.
- Selected e RideReasonSelectedEstimativeID nao devem entrar no treino porque sao campos posteriores ao evento de selecao.

## Conclusao

- O unico sinal linear forte encontrado foi RidePrice, que precisa ser excluido por vazamento.
- Entre variaveis elegiveis, a correlacao linear bruta com o target e baixa; isso sugere que o ganho virá mais de features derivadas e interacoes do que das colunas numericas cruas.
- Coordenadas de origem/destino devem ser condensadas em features espaciais para reduzir redundancia e melhorar interpretabilidade.
- Heatmap salvo em `correlation_heatmap.png`.