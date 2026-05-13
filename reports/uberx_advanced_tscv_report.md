# Modelos Avancados para UberX com TSCV

- Gerado em: `2026-05-13T11:52:02.960423`
- Dataset: `features_uberx.parquet`
- Modelos treinados: `XGBoost`, `LightGBM` e `RandomForest` com hiperparametros padrao.

## Resultado DE

- O mesmo TSCV da etapa baseline foi reutilizado para UberX, preservando comparabilidade temporal.
- As metricas por fold e as medias gerais foram persistidas em CSV, junto com o comparativo contra o melhor baseline anterior.

## Resultado DS

|   CategoryID | DatasetName   | ModelName    |   Folds |   MeanMAE |   StdMAE |   MeanRMSE |   StdRMSE |   MeanMAPE |   StdMAPE |   MeanR2 |     StdR2 |   MeanFeatureCount |
|-------------:|:--------------|:-------------|--------:|----------:|---------:|-----------:|----------:|-----------:|----------:|---------:|----------:|-------------------:|
|            2 | UberX         | LightGBM     |       4 |   6.30405 | 0.749745 |    11.877  |   1.99442 |    18.7329 |   2.40122 | 0.790554 | 0.0753262 |                 35 |
|            2 | UberX         | RandomForest |       4 |   6.33054 | 0.598707 |    12.0872 |   1.6025  |    18.82   |   3.59085 | 0.7832   | 0.0695464 |                 35 |
|            2 | UberX         | XGBoost      |       4 |   9.70292 | 4.33266  |    17.2651 |   7.15312 |    36.5531 |  23.8847  | 0.530016 | 0.366741  |                 35 |

## Comparacao com o Melhor Baseline de UberX

|   CategoryID | DatasetName   | ModelName    |   Folds |   MeanMAE |   StdMAE |   MeanRMSE |   StdRMSE |   MeanMAPE |   StdMAPE |   MeanR2 |     StdR2 |   MeanFeatureCount | BestBaselineModel   |   BestBaselineMAE |   BestBaselineRMSE |   BestBaselineMAPE |   BestBaselineR2 |   DeltaMAEvsBestBaseline |   DeltaRMSEvsBestBaseline |   DeltaMAPEvsBestBaseline |   DeltaR2vsBestBaseline | BeatsBestBaselineMAE   |
|-------------:|:--------------|:-------------|--------:|----------:|---------:|-----------:|----------:|-----------:|----------:|---------:|----------:|-------------------:|:--------------------|------------------:|-------------------:|-------------------:|-----------------:|-------------------------:|--------------------------:|--------------------------:|------------------------:|:-----------------------|
|            2 | UberX         | LightGBM     |       4 |   6.30405 | 0.749745 |    11.877  |   1.99442 |    18.7329 |   2.40122 | 0.790554 | 0.0753262 |                 35 | Lasso               |           7.99423 |            14.7999 |            25.2968 |         0.673356 |                 -1.69018 |                  -2.92296 |                  -6.56392 |                0.117198 | True                   |
|            2 | UberX         | RandomForest |       4 |   6.33054 | 0.598707 |    12.0872 |   1.6025  |    18.82   |   3.59085 | 0.7832   | 0.0695464 |                 35 | Lasso               |           7.99423 |            14.7999 |            25.2968 |         0.673356 |                 -1.66369 |                  -2.71272 |                  -6.47682 |                0.109844 | True                   |
|            2 | UberX         | XGBoost      |       4 |   9.70292 | 4.33266  |    17.2651 |   7.15312 |    36.5531 |  23.8847  | 0.530016 | 0.366741  |                 35 | Lasso               |           7.99423 |            14.7999 |            25.2968 |         0.673356 |                  1.70868 |                   2.46515 |                  11.2563  |               -0.14334  | False                  |

## Leitura DS

- O melhor modelo avancado por MAE foi `LightGBM` com MAE medio `6.3040`, RMSE `11.8770`, MAPE `18.73%` e R2 `0.7906`.
- Comparado ao melhor baseline (`Lasso`), o delta de MAE foi `-1.6902` e o delta de R2 foi `0.1172`.

## Detalhe por Fold

|   CategoryID | DatasetName   | ModelName    | FoldName   |   TrainRows |   EvaluationRows | TrainStart   | TrainEnd   | EvaluationStart   | EvaluationEnd   |   FeatureCount |      MAE |    RMSE |    MAPE |          R2 |
|-------------:|:--------------|:-------------|:-----------|------------:|-----------------:|:-------------|:-----------|:------------------|:----------------|---------------:|---------:|--------:|--------:|------------:|
|            2 | UberX         | LightGBM     | fold_1     |       27535 |             7647 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             35 |  5.81726 | 11.6024 | 21.1313 |  0.764481   |
|            2 | UberX         | LightGBM     | fold_2     |       33773 |             8940 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             35 |  7.41429 | 14.7627 | 19.6202 |  0.695837   |
|            2 | UberX         | LightGBM     | fold_3     |       41871 |            12896 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             35 |  6.09818 | 10.3254 | 18.7273 |  0.85711    |
|            2 | UberX         | LightGBM     | fold_4     |       54095 |            15752 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             35 |  5.88647 | 10.8174 | 15.4529 |  0.844788   |
|            2 | UberX         | RandomForest | fold_1     |       27535 |             7647 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             35 |  6.4577  | 12.538  | 23.6138 |  0.724969   |
|            2 | UberX         | RandomForest | fold_2     |       33773 |             8940 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             35 |  7.11279 | 14.1147 | 19.4864 |  0.721953   |
|            2 | UberX         | RandomForest | fold_3     |       41871 |            12896 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             35 |  5.74589 | 10.4487 | 16.5091 |  0.853676   |
|            2 | UberX         | RandomForest | fold_4     |       54095 |            15752 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             35 |  6.0058  | 11.2475 | 15.6708 |  0.832201   |
|            2 | UberX         | XGBoost      | fold_1     |       27535 |             7647 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             35 |  7.29716 | 13.2017 | 29.0364 |  0.695081   |
|            2 | UberX         | XGBoost      | fold_2     |       33773 |             8940 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             35 |  8.23231 | 16.6392 | 22.5604 |  0.6136     |
|            2 | UberX         | XGBoost      | fold_3     |       41871 |            12896 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             35 |  7.12157 | 11.685  | 22.5307 |  0.817      |
|            2 | UberX         | XGBoost      | fold_4     |       54095 |            15752 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             35 | 16.1606  | 27.5345 | 72.085  | -0.00561805 |