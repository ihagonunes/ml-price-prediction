# Modelos Avancados para Uber Black com TSCV

- Gerado em: `2026-05-13T12:04:31.755503`
- Dataset: `features_black.parquet`
- Modelos treinados: `XGBoost`, `LightGBM` e `RandomForest` com hiperparametros padrao.

## Resultado DE

- O mesmo TSCV da etapa baseline foi reutilizado para Uber Black, preservando comparabilidade temporal.
- As metricas por fold e as medias gerais foram persistidas em CSV, junto com o comparativo contra o melhor baseline anterior.

## Resultado DS

|   CategoryID | DatasetName   | ModelName    |   Folds |   MeanMAE |   StdMAE |   MeanRMSE |   StdRMSE |   MeanMAPE |   StdMAPE |   MeanR2 |      StdR2 |   MeanFeatureCount |
|-------------:|:--------------|:-------------|--------:|----------:|---------:|-----------:|----------:|-----------:|----------:|---------:|-----------:|-------------------:|
|            4 | Uber Black    | LightGBM     |       4 |   6.35687 | 0.441119 |    10.6003 |  0.818884 |    11.4597 |  0.19175  | 0.92679  | 0.00897449 |                 30 |
|            4 | Uber Black    | RandomForest |       4 |   6.44561 | 0.517517 |    11.1014 |  0.685255 |    11.2316 |  0.926514 | 0.919645 | 0.00894276 |                 30 |
|            4 | Uber Black    | XGBoost      |       4 |   6.94036 | 0.590532 |    11.3281 |  0.870054 |    13.2169 |  1.24652  | 0.916315 | 0.0109202  |                 30 |

## Comparacao com o Melhor Baseline de Uber Black

|   CategoryID | DatasetName   | ModelName    |   Folds |   MeanMAE |   StdMAE |   MeanRMSE |   StdRMSE |   MeanMAPE |   StdMAPE |   MeanR2 |      StdR2 |   MeanFeatureCount | BestBaselineModel   |   BestBaselineMAE |   BestBaselineRMSE |   BestBaselineMAPE |   BestBaselineR2 |   DeltaMAEvsBestBaseline |   DeltaRMSEvsBestBaseline |   DeltaMAPEvsBestBaseline |   DeltaR2vsBestBaseline | BeatsBestBaselineMAE   |
|-------------:|:--------------|:-------------|--------:|----------:|---------:|-----------:|----------:|-----------:|----------:|---------:|-----------:|-------------------:|:--------------------|------------------:|-------------------:|-------------------:|-----------------:|-------------------------:|--------------------------:|--------------------------:|------------------------:|:-----------------------|
|            4 | Uber Black    | LightGBM     |       4 |   6.35687 | 0.441119 |    10.6003 |  0.818884 |    11.4597 |  0.19175  | 0.92679  | 0.00897449 |                 30 | LinearRegression    |           9.91222 |            14.6721 |            21.9823 |         0.860094 |                 -3.55535 |                  -4.07175 |                  -10.5226 |               0.0666966 | True                   |
|            4 | Uber Black    | RandomForest |       4 |   6.44561 | 0.517517 |    11.1014 |  0.685255 |    11.2316 |  0.926514 | 0.919645 | 0.00894276 |                 30 | LinearRegression    |           9.91222 |            14.6721 |            21.9823 |         0.860094 |                 -3.46661 |                  -3.57065 |                  -10.7506 |               0.0595514 | True                   |
|            4 | Uber Black    | XGBoost      |       4 |   6.94036 | 0.590532 |    11.3281 |  0.870054 |    13.2169 |  1.24652  | 0.916315 | 0.0109202  |                 30 | LinearRegression    |           9.91222 |            14.6721 |            21.9823 |         0.860094 |                 -2.97186 |                  -3.34401 |                   -8.7654 |               0.056221  | True                   |

## Leitura DS

- O melhor modelo avancado por MAE foi `LightGBM` com MAE medio `6.3569`, RMSE `10.6003`, MAPE `11.46%` e R2 `0.9268`.
- Comparado ao melhor baseline (`LinearRegression`), o delta de MAE foi `-3.5554` e o delta de R2 foi `0.0667`.

## Detalhe por Fold

|   CategoryID | DatasetName   | ModelName    | FoldName   |   TrainRows |   EvaluationRows | TrainStart   | TrainEnd   | EvaluationStart   | EvaluationEnd   |   FeatureCount |     MAE |     RMSE |     MAPE |       R2 |
|-------------:|:--------------|:-------------|:-----------|------------:|-----------------:|:-------------|:-----------|:------------------|:----------------|---------------:|--------:|---------:|---------:|---------:|
|            4 | Uber Black    | LightGBM     | fold_1     |        8052 |             2604 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             30 | 6.2137  | 10.7588  | 11.3762  | 0.924193 |
|            4 | Uber Black    | LightGBM     | fold_2     |       10093 |             3342 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             30 | 5.8475  |  9.48094 | 11.6976  | 0.933629 |
|            4 | Uber Black    | LightGBM     | fold_3     |       12992 |             5873 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             30 | 6.89633 | 11.4495  | 11.5144  | 0.915206 |
|            4 | Uber Black    | LightGBM     | fold_4     |       18541 |             6238 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             30 | 6.46994 | 10.7121  | 11.2505  | 0.934134 |
|            4 | Uber Black    | RandomForest | fold_1     |        8052 |             2604 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             30 | 5.90131 | 10.9244  |  9.92309 | 0.92184  |
|            4 | Uber Black    | RandomForest | fold_2     |       10093 |             3342 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             30 | 6.19246 | 10.3285  | 11.917   | 0.921231 |
|            4 | Uber Black    | RandomForest | fold_3     |       12992 |             5873 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             30 | 7.0945  | 11.9819  | 11.8587  | 0.907137 |
|            4 | Uber Black    | RandomForest | fold_4     |       18541 |             6238 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             30 | 6.59417 | 11.1708  | 11.2278  | 0.928371 |
|            4 | Uber Black    | XGBoost      | fold_1     |        8052 |             2604 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             30 | 6.90763 | 11.7502  | 13.8534  | 0.909578 |
|            4 | Uber Black    | XGBoost      | fold_2     |       10093 |             3342 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             30 | 6.3586  | 10.1357  | 12.9899  | 0.924144 |
|            4 | Uber Black    | XGBoost      | fold_3     |       12992 |             5873 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             30 | 7.75646 | 12.1459  | 14.4475  | 0.904578 |
|            4 | Uber Black    | XGBoost      | fold_4     |       18541 |             6238 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             30 | 6.73875 | 11.2804  | 11.5767  | 0.926959 |