# Analise do Target Price por Categoria

- Gerado em: `2026-05-04T21:55:05.405573`
- Fonte: `c:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical`
- Leitura feita diretamente do Parquet particionado por `CategoryID`, com filtro nas categorias `2`, `9` e `4`.
- Categorias analisadas: `UberX (2)`, `Uber Comfort (9)` e `Uber Black (4)`.

## Resumo Estatistico

|   CategoryID | CategoryName   |   Rows |   FragmentsScanned |    Mean |   Median |     Std |   Min |      Max |   Skewness |   IQR |   MildOutlierLower |   MildOutlierUpper |   ExtremeOutlierLower |   ExtremeOutlierUpper |   MildOutlierCount |   MildOutlierPct |   ExtremeOutlierCount |   ExtremeOutlierPct |   ZeroPriceCount |   ZeroPricePct |
|-------------:|:---------------|-------:|-------------------:|--------:|---------:|--------:|------:|---------:|-----------:|------:|-------------------:|-------------------:|----------------------:|----------------------:|-------------------:|-----------------:|----------------------:|--------------------:|-----------------:|---------------:|
|            2 | UberX          | 710134 |                 20 | 29.5641 |     21   | 81.371  |  0    | 13918.8  |    74.2729 |  21.5 |             -18.75 |              67.25 |                   -51 |                  99.5 |              39122 |             5.51 |                 13250 |                1.87 |                3 |              0 |
|            9 | Uber Comfort   | 274144 |                 20 | 43.6349 |     33.5 | 46.6183 |  5.88 |  8193.38 |    55.5139 |  36   |             -34    |             110    |                   -88 |                 164   |              12632 |             4.61 |                  3047 |                1.11 |                0 |              0 |
|            4 | Uber Black     | 124981 |                 20 | 53.7458 |     44   | 40.43   |  6.03 |  1796.49 |     4.7024 |  47   |             -45.5  |             142.5  |                  -116 |                 213   |               3598 |             2.88 |                   419 |                0.34 |                0 |              0 |

## Percentis

|   CategoryID | CategoryName   |    p1 |    p5 |   p25 |   p50 |   p75 |    p90 |      p95 |   p99 |   p99_5 |   p99_9 |
|-------------:|:---------------|------:|------:|------:|------:|------:|-------:|---------:|------:|--------:|--------:|
|            2 | UberX          |  6.47 |  7.77 |  13.5 |  21   |    35 |  54.22 |  69.6235 | 125   | 163.407 | 281.673 |
|            9 | Uber Comfort   |  8.5  | 10.5  |  20   |  33.5 |    56 |  85    | 108      | 169.5 | 210     | 328.5   |
|            4 | Uber Black     | 11.5  | 13.5  |  25   |  44   |    72 | 104    | 126      | 177.5 | 198.5   | 307.51  |

## Interpretacao DS

### UberX

- Mediana `21.0` | media `29.5641` | desvio `81.371`.
- Assimetria `74.2729`: distribuicao fortemente assimetrica a direita.
- p95 `69.6235` | p99 `125.0` | max `13918.82`.
- Outliers extremos (regra 3*IQR): `13250` linhas (`1.87%`) acima de `99.5`.
- Recomenda-se tratamento robusto antes do treino.

### Uber Comfort

- Mediana `33.5` | media `43.6349` | desvio `46.6183`.
- Assimetria `55.5139`: distribuicao fortemente assimetrica a direita.
- p95 `108.0` | p99 `169.5` | max `8193.38`.
- Outliers extremos (regra 3*IQR): `3047` linhas (`1.11%`) acima de `164.0`.
- Recomenda-se tratamento robusto antes do treino.

### Uber Black

- Mediana `44.0` | media `53.7458` | desvio `40.43`.
- Assimetria `4.7024`: distribuicao fortemente assimetrica a direita.
- p95 `126.0` | p99 `177.5` | max `1796.49`.
- Outliers extremos (regra 3*IQR): `419` linhas (`0.34%`) acima de `213.0`.
- Recomenda-se tratamento robusto antes do treino.

## Graficos

- Histograms: `target_price_histograms.png`
- Boxplots: `target_price_boxplots.png`

## Conclusao

- As tres categorias apresentam cauda a direita e presenca de outliers altos no target.
- Uber Black tende a concentrar valores centrais mais altos e cauda mais longa em termos absolutos.
- Antes do treino, vale testar abordagem robusta para o target, como winsorizacao leve, clipping por regra estatistica ou avaliacao de transformacao log1p em experimentos controlados.