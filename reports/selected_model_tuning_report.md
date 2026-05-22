# Otimizacao de Hiperparametros dos Modelos Selecionados

- Gerado em: `2026-05-20T17:07:57.947419`
- Metodo: `Optuna TPE` com `15` trials por categoria.
- Modelo otimizado: `LightGBM` nas categorias `UberX`, `Uber Comfort` e `Uber Black`.
- Criterio objetivo: `menor RMSE medio no TSCV`.

## Resumo de Ganho

|   CategoryID | DatasetName   |   BestTrialNumber |   TrialsExecuted |   DefaultMeanRMSE |   TunedMeanRMSE |   DeltaRMSE |   DefaultMeanMAE |   TunedMeanMAE |    DeltaMAE |   DefaultMeanMAPE |   TunedMeanMAPE |   DeltaMAPE |   DefaultMeanR2 |   TunedMeanR2 |     DeltaR2 |
|-------------:|:--------------|------------------:|-----------------:|------------------:|----------------:|------------:|-----------------:|---------------:|------------:|------------------:|----------------:|------------:|----------------:|--------------:|------------:|
|            2 | UberX         |                12 |               15 |          11.877   |        11.432   |  -0.444981  |          6.30405 |        5.92626 | -0.377795   |           18.7329 |        17.5988  |   -1.13411  |        0.790554 |      0.807715 | 0.0171608   |
|            4 | Uber Black    |                11 |               15 |          10.6003  |        10.2841  |  -0.316183  |          6.35687 |        6.34965 | -0.00722094 |           11.4597 |        11.8639  |    0.404271 |        0.92679  |      0.931019 | 0.00422824  |
|            9 | Uber Comfort  |                12 |               15 |           7.46738 |         7.42262 |  -0.0447582 |          4.14988 |        4.1968  |  0.0469149  |            8.3347 |         8.69247 |    0.357774 |        0.957665 |      0.958323 | 0.000657716 |

## Melhores Hiperparametros

### UberX

- Melhor trial: `12` de `15`.
- RMSE default vs tuned: `11.8770` -> `11.4320` (delta `-0.4450`).
- MAE default vs tuned: `6.3040` -> `5.9263` (delta `-0.3778`).
- MAPE default vs tuned: `18.73%` -> `17.60%` (delta `-1.13`).
- R2 default vs tuned: `0.7906` -> `0.8077` (delta `0.0172`).
- Hiperparametros: `{"bagging_fraction": 0.85, "bagging_freq": 5, "deterministic": true, "feature_fraction": 0.6, "force_col_wise": true, "lambda_l1": 1.06306407977467e-08, "lambda_l2": 9.868745464853728, "learning_rate": 0.02714108732037047, "max_depth": -1, "metric": "l2", "min_data_in_leaf": 70, "min_gain_to_split": 0.004536471339817827, "num_boost_round": 350, "num_leaves": 99, "num_threads": -1, "objective": "regression", "seed": 42, "verbosity": -1}`.

### Uber Black

- Melhor trial: `11` de `15`.
- RMSE default vs tuned: `10.6003` -> `10.2841` (delta `-0.3162`).
- MAE default vs tuned: `6.3569` -> `6.3496` (delta `-0.0072`).
- MAPE default vs tuned: `11.46%` -> `11.86%` (delta `0.40`).
- R2 default vs tuned: `0.9268` -> `0.9310` (delta `0.0042`).
- Hiperparametros: `{"bagging_fraction": 1.0, "bagging_freq": 0, "deterministic": true, "feature_fraction": 0.7, "force_col_wise": true, "lambda_l1": 0.00021535541293642255, "lambda_l2": 0.13241042802392494, "learning_rate": 0.08255949675641235, "max_depth": 4, "metric": "l2", "min_data_in_leaf": 60, "min_gain_to_split": 0.09736015645089524, "num_boost_round": 350, "num_leaves": 18, "num_threads": -1, "objective": "regression", "seed": 42, "verbosity": -1}`.

### Uber Comfort

- Melhor trial: `12` de `15`.
- RMSE default vs tuned: `7.4674` -> `7.4226` (delta `-0.0448`).
- MAE default vs tuned: `4.1499` -> `4.1968` (delta `0.0469`).
- MAPE default vs tuned: `8.33%` -> `8.69%` (delta `0.36`).
- R2 default vs tuned: `0.9577` -> `0.9583` (delta `0.0007`).
- Hiperparametros: `{"bagging_fraction": 0.9, "bagging_freq": 0, "deterministic": true, "feature_fraction": 0.6, "force_col_wise": true, "lambda_l1": 9.265701961309923e-05, "lambda_l2": 9.868745464853728, "learning_rate": 0.10439338960876268, "max_depth": -1, "metric": "l2", "min_data_in_leaf": 70, "min_gain_to_split": 0.9845823199357809, "num_boost_round": 300, "num_leaves": 44, "num_threads": -1, "objective": "regression", "seed": 42, "verbosity": -1}`.

## Leitura DS

- O ganho principal foi avaliado por RMSE medio, que continua sendo a metrica de selecao adotada no projeto.
- Melhorias simultaneas em RMSE e MAE indicam calibracao mais robusta do erro absoluto; quando o ganho em MAPE for menor, isso sugere que a distribuicao relativa do erro ja estava bem capturada no modelo padrao.
- Se alguma categoria mostrar ganho marginal, isso sinaliza que o modelo default ja estava perto de um bom regime e que o proximo salto pode depender mais de novas features do que de tuning adicional.
