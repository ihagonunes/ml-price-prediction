# Estrategia Temporal de Train/Test e TSCV

- Gerado em: `2026-05-20T16:37:05.155947`
- Fonte: `C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\data\analytical_curated`
- Regime usado para modelagem: `2021-11-01` ate `2022-06-14`

## Decisao DS

- O desenho parte do regime mais recente (`2021-11-01` em diante), porque a analise temporal mostrou quebra forte de volume entre `ago-out/2021` e `nov/2021-jun/2022`.
- O holdout final usa os ultimos `28` dias (`2022-05-18` a `2022-06-14`), com embargo de `7` dias para evitar leakage temporal entre treino e teste.
- O TSCV interno usa `4` folds expansivos, cada um com `28` dias de validacao e `gap` de `7` dias.
- A ancora temporal e `Create`, nao `Updated`, e o split e aplicado no nivel de dia para respeitar semanas completas e manter todas as estimativas da mesma corrida no mesmo lado do corte.

## Resumo Geral dos Splits

| split_name   | split_type   | category_id   | train_start   | train_end   | gap_start   | gap_end    | evaluation_start   | evaluation_end   |   train_days |   gap_days |   evaluation_days |   train_rows |   evaluation_rows |   train_unique_rides |   evaluation_unique_rides |   ride_overlap |
|:-------------|:-------------|:--------------|:--------------|:------------|:------------|:-----------|:-------------------|:-----------------|-------------:|-----------:|------------------:|-------------:|------------------:|---------------------:|--------------------------:|---------------:|
| fold_1       | cv           |               | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |           72 |          7 |                28 |       104318 |             38660 |                12735 |                      3867 |              0 |
| fold_2       | cv           |               | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |          100 |          7 |                28 |       135617 |             44373 |                15885 |                      4200 |              0 |
| fold_3       | cv           |               | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |          128 |          7 |                28 |       175178 |             72428 |                19594 |                      6935 |              0 |
| fold_4       | cv           |               | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |          156 |          7 |                28 |       242864 |             83632 |                26169 |                      7525 |              0 |
| holdout_test | holdout      |               | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |          191 |          7 |                28 |       347134 |             90668 |                35626 |                      8228 |              0 |

## Volume das Categorias-Alvo

| split_name   | split_type   |   category_id | train_start   | train_end   | gap_start   | gap_end    | evaluation_start   | evaluation_end   |   train_days |   gap_days |   evaluation_days |   train_rows |   evaluation_rows |   train_unique_rides |   evaluation_unique_rides |   ride_overlap |
|:-------------|:-------------|--------------:|:--------------|:------------|:------------|:-----------|:-------------------|:-----------------|-------------:|-----------:|------------------:|-------------:|------------------:|---------------------:|--------------------------:|---------------:|
| fold_1       | cv           |             2 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |           72 |          7 |                28 |        27535 |              7647 |                12731 |                      3867 |              0 |
| fold_1       | cv           |             9 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |           71 |          7 |                28 |        14563 |              5536 |                10294 |                      3396 |              0 |
| fold_1       | cv           |             4 | 2021-11-01    | 2022-01-11  | 2022-01-12  | 2022-01-18 | 2022-01-19         | 2022-02-15       |           67 |          7 |                28 |         8052 |              2604 |                 7742 |                      2552 |              0 |
| fold_2       | cv           |             2 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |          100 |          7 |                28 |        33773 |              8940 |                15881 |                      4200 |              0 |
| fold_2       | cv           |             9 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |           99 |          7 |                28 |        18980 |              6580 |                13017 |                      4101 |              0 |
| fold_2       | cv           |             4 | 2021-11-01    | 2022-02-08  | 2022-02-09  | 2022-02-15 | 2022-02-16         | 2022-03-15       |           95 |          7 |                28 |        10093 |              3342 |                 9714 |                      3253 |              0 |
| fold_3       | cv           |             2 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |          128 |          7 |                28 |        41871 |             12896 |                19590 |                      6935 |              0 |
| fold_3       | cv           |             9 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |          127 |          7 |                28 |        24772 |             10979 |                16593 |                      6832 |              0 |
| fold_3       | cv           |             4 | 2021-11-01    | 2022-03-08  | 2022-03-09  | 2022-03-15 | 2022-03-16         | 2022-04-12       |          123 |          7 |                28 |        12992 |              5873 |                12535 |                      5805 |              0 |
| fold_4       | cv           |             2 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |          156 |          7 |                28 |        54095 |             15752 |                26165 |                      7525 |              0 |
| fold_4       | cv           |             9 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |          155 |          7 |                28 |        35107 |             11063 |                23054 |                      7369 |              0 |
| fold_4       | cv           |             4 | 2021-11-01    | 2022-04-05  | 2022-04-06  | 2022-04-12 | 2022-04-13         | 2022-05-10       |          151 |          7 |                28 |        18541 |              6238 |                17997 |                      6099 |              0 |
| holdout_test | holdout      |             2 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |          191 |          7 |                28 |        73490 |             16456 |                35622 |                      8226 |              0 |
| holdout_test | holdout      |             9 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |          190 |          7 |                28 |        49289 |             12038 |                32327 |                      8035 |              0 |
| holdout_test | holdout      |             4 | 2021-11-01    | 2022-05-10  | 2022-05-11  | 2022-05-17 | 2022-05-18         | 2022-06-14       |          186 |          7 |                28 |        26385 |              7004 |                25694 |                      6839 |              0 |

## Validacao de Integridade

- Overlap de `RideID` entre treino e validacao/teste: `0` em todos os splits.
- Cobertura disponivel no regime escolhido: `226` dias, `463158` linhas e `46052` corridas unicas.
- Categoria `2` no holdout: 16456 linhas e 8226 corridas unicas.
- Categoria `9` no holdout: 12038 linhas e 8035 corridas unicas.
- Categoria `4` no holdout: 7004 linhas e 6839 corridas unicas.

## Reuso nos 3 Pipelines

- `iter_tscv_splits(...)` expone os folds internos e `get_holdout_split(...)` retorna o corte final, com suporte opcional a filtro por `CategoryID`.
- Isso permite que os tres pipelines de modelagem usem exatamente as mesmas janelas temporais, mantendo comparabilidade entre experimentos.