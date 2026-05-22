# Modelos Avancados para Uber Comfort com TSCV

- Gerado em: `2026-05-20T17:01:55.629804`
- Dataset: `features_comfort.parquet`
- Modelos treinados: `XGBoost`, `LightGBM` e `RandomForest` com hiperparametros padrao.

## Resultado DE

- O mesmo TSCV da etapa baseline foi reutilizado para Uber Comfort, preservando comparabilidade temporal.
- As metricas por fold e as medias gerais foram persistidas em CSV, junto com o comparativo contra o melhor baseline anterior.

## Resultado DS

|   CategoryID | DatasetName   | ModelName    |   Folds |   MeanMAE |   StdMAE |   MeanRMSE |   StdRMSE |   MeanMAPE |   StdMAPE |   MeanR2 |     StdR2 |   MeanFeatureCount |
|-------------:|:--------------|:-------------|--------:|----------:|---------:|-----------:|----------:|-----------:|----------:|---------:|----------:|-------------------:|
|            9 | Uber Comfort  | LightGBM     |       4 |   4.14988 | 0.564189 |    7.46738 |   1.1657  |    8.3347  |  0.474051 | 0.957665 | 0.0120929 |                 31 |
|            9 | Uber Comfort  | RandomForest |       4 |   4.19764 | 0.519762 |    7.90239 |   1.01657 |    8.26741 |  0.692277 | 0.952743 | 0.0110361 |                 31 |
|            9 | Uber Comfort  | XGBoost      |       4 |   4.32068 | 0.730102 |    7.59965 |   1.09855 |    8.99602 |  1.38973  | 0.956156 | 0.0121162 |                 31 |

## Comparacao com o Melhor Baseline de Uber Comfort

|   CategoryID | DatasetName   | ModelName    |   Folds |   MeanMAE |   StdMAE |   MeanRMSE |   StdRMSE |   MeanMAPE |   StdMAPE |   MeanR2 |     StdR2 |   MeanFeatureCount | BestBaselineModel   |   BestBaselineMAE |   BestBaselineRMSE |   BestBaselineMAPE |   BestBaselineR2 |   DeltaMAEvsBestBaseline |   DeltaRMSEvsBestBaseline |   DeltaMAPEvsBestBaseline |   DeltaR2vsBestBaseline | BeatsBestBaselineMAE   |
|-------------:|:--------------|:-------------|--------:|----------:|---------:|-----------:|----------:|-----------:|----------:|---------:|----------:|-------------------:|:--------------------|------------------:|-------------------:|-------------------:|-----------------:|-------------------------:|--------------------------:|--------------------------:|------------------------:|:-----------------------|
|            9 | Uber Comfort  | LightGBM     |       4 |   4.14988 | 0.564189 |    7.46738 |   1.1657  |    8.3347  |  0.474051 | 0.957665 | 0.0120929 |                 31 | Lasso               |           5.59615 |            9.20783 |            12.8483 |         0.936263 |                 -1.44626 |                  -1.74045 |                  -4.51364 |               0.021402  | True                   |
|            9 | Uber Comfort  | RandomForest |       4 |   4.19764 | 0.519762 |    7.90239 |   1.01657 |    8.26741 |  0.692277 | 0.952743 | 0.0110361 |                 31 | Lasso               |           5.59615 |            9.20783 |            12.8483 |         0.936263 |                 -1.39851 |                  -1.30544 |                  -4.58093 |               0.0164797 | True                   |
|            9 | Uber Comfort  | XGBoost      |       4 |   4.32068 | 0.730102 |    7.59965 |   1.09855 |    8.99602 |  1.38973  | 0.956156 | 0.0121162 |                 31 | Lasso               |           5.59615 |            9.20783 |            12.8483 |         0.936263 |                 -1.27546 |                  -1.60818 |                  -3.85232 |               0.0198931 | True                   |

## Leitura DS

- O melhor modelo avancado por MAE foi `LightGBM` com MAE medio `4.1499`, RMSE `7.4674`, MAPE `8.33%` e R2 `0.9577`.
- Comparado ao melhor baseline (`Lasso`), o delta de MAE foi `-1.4463` e o delta de R2 foi `0.0214`.

## Detalhe por Fold

|   CategoryID | DatasetName   | ModelName    | FoldName   |   TrainRows |   EvaluationRows | TrainStart   | TrainEnd   | EvaluationStart   | EvaluationEnd   |   FeatureCount |     MAE |    RMSE |     MAPE |       R2 |
|-------------:|:--------------|:-------------|:-----------|------------:|-----------------:|:-------------|:-----------|:------------------|:----------------|---------------:|--------:|--------:|---------:|---------:|
|            9 | Uber Comfort  | LightGBM     | fold_1     |       14563 |             5536 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             31 | 3.55759 | 6.16233 |  8.14432 | 0.969995 |
|            9 | Uber Comfort  | LightGBM     | fold_2     |       18980 |             6580 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             31 | 3.98424 | 7.04649 |  8.48764 | 0.961281 |
|            9 | Uber Comfort  | LightGBM     | fold_3     |       24772 |            10979 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             31 | 4.90928 | 8.92484 |  8.90742 | 0.941134 |
|            9 | Uber Comfort  | LightGBM     | fold_4     |       35107 |            11063 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             31 | 4.14841 | 7.73587 |  7.79941 | 0.95825  |
|            9 | Uber Comfort  | RandomForest | fold_1     |       14563 |             5536 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             31 | 3.62907 | 6.62422 |  7.83906 | 0.965328 |
|            9 | Uber Comfort  | RandomForest | fold_2     |       18980 |             6580 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             31 | 4.11541 | 7.88455 |  8.63867 | 0.951524 |
|            9 | Uber Comfort  | RandomForest | fold_3     |       24772 |            10979 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             31 | 4.88955 | 9.11003 |  9.04255 | 0.938666 |
|            9 | Uber Comfort  | RandomForest | fold_4     |       35107 |            11063 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             31 | 4.15653 | 7.99077 |  7.54936 | 0.955454 |
|            9 | Uber Comfort  | XGBoost      | fold_1     |       14563 |             5536 | 2021-11-01   | 2022-01-11 | 2022-01-19        | 2022-02-15      |             31 | 3.72716 | 6.46383 |  8.57652 | 0.966987 |
|            9 | Uber Comfort  | XGBoost      | fold_2     |       18980 |             6580 | 2021-11-01   | 2022-02-08 | 2022-02-16        | 2022-03-15      |             31 | 4.06147 | 7.31104 |  8.53911 | 0.958319 |
|            9 | Uber Comfort  | XGBoost      | fold_3     |       24772 |            10979 | 2021-11-01   | 2022-03-08 | 2022-03-16        | 2022-04-12      |             31 | 5.38584 | 9.0969  | 11.0194  | 0.938842 |
|            9 | Uber Comfort  | XGBoost      | fold_4     |       35107 |            11063 | 2021-11-01   | 2022-04-05 | 2022-04-13        | 2022-05-10      |             31 | 4.10826 | 7.52684 |  7.8491  | 0.960476 |