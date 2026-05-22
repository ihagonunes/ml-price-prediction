# Comparacao de Algoritmos e Selecao por Categoria

- Gerado em: `2026-05-20T17:02:59.649108`
- Criterio primario de selecao: `menor RMSE medio no TSCV`.
- Modelos comparados: `LinearRegression`, `Ridge`, `Lasso`, `XGBoost`, `LightGBM` e `RandomForest`.

## Tabela Consolidada

| CategoryName   | AlgorithmGroup   | ModelName        |   MeanRMSE |   MeanMAE |   MeanMAPE |   MeanR2 |   RMSECategoryRank |
|:---------------|:-----------------|:-----------------|-----------:|----------:|-----------:|---------:|-------------------:|
| UberX          | advanced         | LightGBM         |   11.877   |   6.30405 |   18.7329  | 0.790554 |                  1 |
| UberX          | advanced         | RandomForest     |   12.0844  |   6.33014 |   18.8174  | 0.783347 |                  2 |
| UberX          | baseline         | Lasso            |   14.7999  |   7.99423 |   25.2968  | 0.673356 |                  3 |
| UberX          | baseline         | Ridge            |   14.8247  |   8.01364 |   25.3865  | 0.672122 |                  4 |
| UberX          | baseline         | LinearRegression |   14.8292  |   8.01538 |   25.3911  | 0.671893 |                  5 |
| UberX          | advanced         | XGBoost          |   17.2651  |   9.70292 |   36.5531  | 0.530016 |                  6 |
| Uber Black     | advanced         | LightGBM         |   10.6003  |   6.35687 |   11.4597  | 0.92679  |                  1 |
| Uber Black     | advanced         | RandomForest     |   11.1014  |   6.44561 |   11.2316  | 0.919645 |                  2 |
| Uber Black     | advanced         | XGBoost          |   11.3281  |   6.94036 |   13.2169  | 0.916315 |                  3 |
| Uber Black     | baseline         | Ridge            |   14.6429  |   9.91333 |   22.0572  | 0.860664 |                  4 |
| Uber Black     | baseline         | Lasso            |   14.6633  |   9.91374 |   22.0153  | 0.860262 |                  5 |
| Uber Black     | baseline         | LinearRegression |   14.6721  |   9.91222 |   21.9823  | 0.860094 |                  6 |
| Uber Comfort   | advanced         | LightGBM         |    7.46738 |   4.14988 |    8.3347  | 0.957665 |                  1 |
| Uber Comfort   | advanced         | XGBoost          |    7.59965 |   4.32068 |    8.99602 | 0.956156 |                  2 |
| Uber Comfort   | advanced         | RandomForest     |    7.90239 |   4.19764 |    8.26741 | 0.952743 |                  3 |
| Uber Comfort   | baseline         | LinearRegression |    9.2034  |   5.6052  |   12.8896  | 0.936363 |                  4 |
| Uber Comfort   | baseline         | Ridge            |    9.20666 |   5.60316 |   12.8806  | 0.936303 |                  5 |
| Uber Comfort   | baseline         | Lasso            |    9.20783 |   5.59615 |   12.8483  | 0.936263 |                  6 |

## Melhor Modelo por Categoria

| CategoryName   | SelectedModel   | SelectedModelGroup   |   SelectedMeanRMSE |   SelectedMeanMAE |   SelectedMeanMAPE |   SelectedMeanR2 | BestBaselineRMSEModel   |   BestBaselineRMSE |   DeltaRMSEvsBestBaseline |
|:---------------|:----------------|:---------------------|-------------------:|------------------:|-------------------:|-----------------:|:------------------------|-------------------:|--------------------------:|
| UberX          | LightGBM        | advanced             |           11.877   |           6.30405 |            18.7329 |         0.790554 | Lasso                   |            14.7999 |                  -2.92296 |
| Uber Black     | LightGBM        | advanced             |           10.6003  |           6.35687 |            11.4597 |         0.92679  | Ridge                   |            14.6429 |                  -4.04257 |
| Uber Comfort   | LightGBM        | advanced             |            7.46738 |           4.14988 |             8.3347 |         0.957665 | LinearRegression        |             9.2034 |                  -1.73602 |

## Justificativa DS

### UberX

- Modelo selecionado: `LightGBM` (advanced).
- Escolha por RMSE: `11.8770` vs melhor baseline por RMSE `Lasso` = `14.7999` (delta `-2.9230`).
- Leitura complementar: MAE `6.3040`, MAPE `18.73%`, R2 `0.7906`.
- Trade-offs: Runner-up por RMSE: RandomForest (12.0844).

### Uber Black

- Modelo selecionado: `LightGBM` (advanced).
- Escolha por RMSE: `10.6003` vs melhor baseline por RMSE `Ridge` = `14.6429` (delta `-4.0426`).
- Leitura complementar: MAE `6.3569`, MAPE `11.46%`, R2 `0.9268`.
- Trade-offs: MAPE melhor em RandomForest (11.23%) | Runner-up por RMSE: RandomForest (11.1014).

### Uber Comfort

- Modelo selecionado: `LightGBM` (advanced).
- Escolha por RMSE: `7.4674` vs melhor baseline por RMSE `LinearRegression` = `9.2034` (delta `-1.7360`).
- Leitura complementar: MAE `4.1499`, MAPE `8.33%`, R2 `0.9577`.
- Trade-offs: MAPE melhor em RandomForest (8.27%) | Runner-up por RMSE: XGBoost (7.5997).

## Conclusao

- `LightGBM` foi o melhor modelo por RMSE nas tres categorias-alvo.
- `RandomForest` apareceu como alternativa forte em MAPE para `Uber Comfort` e `Uber Black`, mas nao venceu no criterio principal de selecao.
- Os baselines lineares continuam como piso de comparacao, mas os modelos avancados passaram esse piso com folga em todas as categorias.
