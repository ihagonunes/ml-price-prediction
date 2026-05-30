# Results Comparison

- Gerado em: `2026-05-29T20:53:51.268079`
- Arquivo principal: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\reports\results_comparison.csv`
- Linhas por fold: `84`
- Linhas de media final: `21`
- Categorias avaliadas: `3`
- Algoritmos por categoria: `7` a `7`

## Melhores Modelos por RMSE Medio

|   CategoryID | DatasetName   | Algorithm      | AlgorithmGroup   | ModelVariant   |     RMSE |     MAE |     MAPE |       R2 |
|-------------:|:--------------|:---------------|:-----------------|:---------------|---------:|--------:|---------:|---------:|
|            2 | UberX         | LightGBM_Tuned | tuned            | optimized      | 11.432   | 5.92626 | 17.5988  | 0.807715 |
|            4 | Uber Black    | LightGBM_Tuned | tuned            | optimized      | 10.3364  | 6.40856 | 12.0789  | 0.930395 |
|            9 | Uber Comfort  | LightGBM_Tuned | tuned            | optimized      |  7.42262 | 4.1968  |  8.69247 | 0.958323 |

## Validacao

- O arquivo consolida baselines, modelos avancados padrao e `LightGBM_Tuned`.
- As linhas `fold` preservam as metricas originais de cada fold do TSCV.
- As linhas `mean` calculam as medias finais e os desvios padrao das metricas por categoria e algoritmo.
