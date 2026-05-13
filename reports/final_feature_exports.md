# Exportacao Final dos Datasets de Features por Categoria

- Gerado em: `2026-05-13T08:10:36.811157`
- Fonte: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\features_temporal`
- Saida: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\final_features`

## Resultado DE

- Foram gerados tres arquivos Parquet finais, um por categoria-alvo, prontos para modelagem e TSCV.
- Cada arquivo preserva o target `Price`, a coluna temporal `Create` e a coluna derivada `CreateDate` para ordenacao.
- Colunas de leakage alto identificadas na etapa 3.5 foram removidas da camada final (`Updated*`, `PriceWasCapped` e o preco cruzado da propria categoria).

## Validacao DS

|   CategoryID | DatasetName   | OutputFile                                                                                                                        |   rows_written |   column_count |   engineered_feature_count | target_present   |   target_nulls | time_column_present   | time_order_column_present   |   time_nulls | min_create                    | max_create                    | own_cross_removed   | aux_cross_1   | aux_cross_1_present   | aux_cross_2   | aux_cross_2_present   |   target_positive_rows |
|-------------:|:--------------|:----------------------------------------------------------------------------------------------------------------------------------|---------------:|---------------:|---------------------------:|:-----------------|---------------:|:----------------------|:----------------------------|-------------:|:------------------------------|:------------------------------|:--------------------|:--------------|:----------------------|:--------------|:----------------------|-----------------------:|
|            2 | UberX         | C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\final_features\features_uberx.parquet   |         710046 |             57 |                         23 | True             |              0 | True                  | True                        |            0 | 2021-08-17T10:09:45.642582500 | 2022-06-14T20:56:40.757298900 | True                | Price_Comfort | True                  | Price_Black   | True                  |                 710046 |
|            4 | Uber Black    | C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\final_features\features_black.parquet   |         124981 |             57 |                         23 | True             |              0 | True                  | True                        |            0 | 2021-08-17T10:10:30.822922500 | 2022-06-14T20:56:40.757298900 | True                | Price_UberX   | True                  | Price_Comfort | True                  |                 124981 |
|            9 | Uber Comfort  | C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\final_features\features_comfort.parquet |         274144 |             57 |                         23 | True             |              0 | True                  | True                        |            0 | 2021-08-17T10:09:45.642582500 | 2022-06-14T20:56:40.757298900 | True                | Price_UberX   | True                  | Price_Black   | True                  |                 274144 |

## Conclusao

- Os volumes por categoria estao coerentes com a camada `features_temporal`.
- O target `Price` esta presente em todos os arquivos, sem nulos.
- As features engenheiradas esperadas foram mantidas, incluindo temporais, historico por usuario e precos cruzados auxiliares.