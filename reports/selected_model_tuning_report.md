# Otimizacao de Hiperparametros dos Modelos Selecionados

- Gerado em: `2026-05-13T12:32:13.170464`
- Metodo: `Optuna TPE` com `15` trials por categoria.
- Modelo otimizado: `LightGBM` nas categorias `UberX`, `Uber Comfort` e `Uber Black`.
- Criterio objetivo: `menor RMSE medio no TSCV`.

## Resumo de Ganho

|   CategoryID | DatasetName   |   BestTrialNumber |   TrialsExecuted |   DefaultMeanRMSE |   TunedMeanRMSE |   DeltaRMSE |   DefaultMeanMAE |   TunedMeanMAE |   DeltaMAE |   DefaultMeanMAPE |   TunedMeanMAPE |   DeltaMAPE |   DefaultMeanR2 |   TunedMeanR2 |     DeltaR2 |
|-------------:|:--------------|------------------:|-----------------:|------------------:|----------------:|------------:|-----------------:|---------------:|-----------:|------------------:|----------------:|------------:|----------------:|--------------:|------------:|
|            2 | UberX         |                12 |               15 |          11.877   |        11.432   |  -0.444981  |          6.30405 |        5.92626 | -0.377795  |           18.7329 |        17.5988  |   -1.13411  |        0.790554 |      0.807715 | 0.0171608   |
|            4 | Uber Black    |                13 |               15 |          10.6003  |        10.3364  |  -0.263923  |          6.35687 |        6.40856 |  0.0516951 |           11.4597 |        12.0789  |    0.619264 |        0.92679  |      0.930395 | 0.00360489  |
|            9 | Uber Comfort  |                12 |               15 |           7.46738 |         7.42262 |  -0.0447582 |          4.14988 |        4.1968  |  0.0469149 |            8.3347 |         8.69247 |    0.357774 |        0.957665 |      0.958323 | 0.000657716 |

## Melhores Hiperparametros

### UberX

- Melhor trial: `12` de `15`.
- RMSE default vs tuned: `11.8770` -> `11.4320` (delta `-0.4450`).
- MAE default vs tuned: `6.3040` -> `5.9263` (delta `-0.3778`).
- MAPE default vs tuned: `18.73%` -> `17.60%` (delta `-1.13`).
- R2 default vs tuned: `0.7906` -> `0.8077` (delta `0.0172`).
- Hiperparametros: `{"bagging_fraction": 0.85, "bagging_freq": 5, "deterministic": true, "feature_fraction": 0.6, "force_col_wise": true, "lambda_l1": 1.0630640797746661e-08, "lambda_l2": 9.86874546485385, "learning_rate": 0.027141087320370485, "max_depth": -1, "metric": "l2", "min_data_in_leaf": 70, "min_gain_to_split": 0.004536471339817813, "num_boost_round": 350, "num_leaves": 99, "num_threads": -1, "objective": "regression", "seed": 42, "verbosity": -1}`.

### Uber Black

- Melhor trial: `13` de `15`.
- RMSE default vs tuned: `10.6003` -> `10.3364` (delta `-0.2639`).
- MAE default vs tuned: `6.3569` -> `6.4086` (delta `0.0517`).
- MAPE default vs tuned: `11.46%` -> `12.08%` (delta `0.62`).
- R2 default vs tuned: `0.9268` -> `0.9304` (delta `0.0036`).
- Hiperparametros: `{"bagging_fraction": 1.0, "bagging_freq": 0, "deterministic": true, "feature_fraction": 0.6, "force_col_wise": true, "lambda_l1": 9.990974531593681e-05, "lambda_l2": 5.43332313833525, "learning_rate": 0.10537327151385636, "max_depth": 4, "metric": "l2", "min_data_in_leaf": 70, "min_gain_to_split": 0.036570454874058994, "num_boost_round": 400, "num_leaves": 51, "num_threads": -1, "objective": "regression", "seed": 42, "verbosity": -1}`.

### Uber Comfort

- Melhor trial: `12` de `15`.
- RMSE default vs tuned: `7.4674` -> `7.4226` (delta `-0.0448`).
- MAE default vs tuned: `4.1499` -> `4.1968` (delta `0.0469`).
- MAPE default vs tuned: `8.33%` -> `8.69%` (delta `0.36`).
- R2 default vs tuned: `0.9577` -> `0.9583` (delta `0.0007`).
- Hiperparametros: `{"bagging_fraction": 0.9, "bagging_freq": 0, "deterministic": true, "feature_fraction": 0.6, "force_col_wise": true, "lambda_l1": 9.265701961309923e-05, "lambda_l2": 9.86874546485385, "learning_rate": 0.10439338960876268, "max_depth": -1, "metric": "l2", "min_data_in_leaf": 70, "min_gain_to_split": 0.9845823199357808, "num_boost_round": 300, "num_leaves": 44, "num_threads": -1, "objective": "regression", "seed": 42, "verbosity": -1}`.

## Leitura DS

- O ganho principal foi avaliado por RMSE medio, que continua sendo a metrica de selecao adotada no projeto.
- Melhorias simultaneas em RMSE e MAE indicam calibracao mais robusta do erro absoluto; quando o ganho em MAPE for menor, isso sugere que a distribuicao relativa do erro ja estava bem capturada no modelo padrao.
- Se alguma categoria mostrar ganho marginal, isso sinaliza que o modelo default ja estava perto de um bom regime e que o proximo salto pode depender mais de novas features do que de tuning adicional.
