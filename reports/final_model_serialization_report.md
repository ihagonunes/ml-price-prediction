# Serializacao dos Modelos Finais

- Gerado em: `2026-05-28T20:52:01.102221`
- Diretorio dos modelos: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\models`
- Formato: `.joblib` com artefato contendo preprocessor, booster LightGBM, lista de features e metadados.

## Validacao de Reload e Predicao

|   CategoryID | DatasetName   | ModelName   | ModelPath                                                                                                        | ModelFileName        |   ModelFileSizeMB |   TrainingRows |   FeatureCount | TrainingStartDate   | TrainingEndDate   |   ExampleRows |   ExampleTargetPrice |   ExamplePrediction | PredictionIsFinite   | LoadedArtifactClass   | ValidatedAt                |
|-------------:|:--------------|:------------|:-----------------------------------------------------------------------------------------------------------------|:---------------------|------------------:|---------------:|---------------:|:--------------------|:------------------|--------------:|---------------------:|--------------------:|:---------------------|:----------------------|:---------------------------|
|            2 | UberX         | LightGBM    | C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\models\model_uberx.joblib   | model_uberx.joblib   |            1.0722 |          94749 |             35 | 2021-11-01          | 2022-06-14        |             1 |                58.73 |             48.1453 | True                 | FinalLightGBMModel    | 2026-05-28T20:51:54.016115 |
|            9 | Uber Comfort  | LightGBM    | C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\models\model_comfort.joblib | model_comfort.joblib |            0.4093 |          64586 |             31 | 2021-11-01          | 2022-06-14        |             1 |               133    |            118.1    | True                 | FinalLightGBMModel    | 2026-05-28T20:51:59.105558 |
|            4 | Uber Black    | LightGBM    | C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\models\model_black.joblib   | model_black.joblib   |            0.1659 |          35235 |             30 | 2021-11-01          | 2022-06-14        |             1 |                69.5  |            103.596  | True                 | FinalLightGBMModel    | 2026-05-28T20:52:01.054628 |

## Resultado

- Os tres modelos finais otimizados foram serializados em `/models/`.
- Cada arquivo foi recarregado com `joblib.load` e gerou uma predicao finita a partir de um exemplo real de input.
- Os arquivos `.joblib` permanecem ignorados pelo Git; o script versionavel recria os artefatos quando necessario.
