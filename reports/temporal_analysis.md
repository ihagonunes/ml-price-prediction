# Analise Temporal dos Dados

- Gerado em: `2026-05-04T22:29:27.874538`
- Fonte: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical`
- Escopo: `ride`, `rideestimative` e `product` reconstruidos a partir do Parquet consolidado.

## Cobertura Temporal

| table_name     | time_field   |   row_count | min_timestamp                 | max_timestamp                 | null_count   | distinct_dates   | note                                                      |
|:---------------|:-------------|------------:|:------------------------------|:------------------------------|:-------------|:-----------------|:----------------------------------------------------------|
| ride           | Create       |      239270 | 2021-08-17 10:09:45.642582500 | 2022-06-14 20:56:40.757298900 | 0            | 302              | Campo nativo da tabela ride.                              |
| ride           | Schedule     |      239270 | 2021-08-17 10:09:45.628170900 | 2022-06-14 20:56:40.699043200 | 0            | 302              | Campo nativo da tabela ride.                              |
| ride           | Updated      |      239270 | 2021-08-17 10:09:53.467805100 | 2022-06-14 20:56:55.425096800 | 0            | 302              | Campo nativo da tabela ride.                              |
| rideestimative | Create       |     2000000 | 2021-08-17 10:09:45.642582500 | 2022-06-14 20:56:40.757298900 | 0            | 302              | Cobertura herdada do timestamp da corrida associada.      |
| rideestimative | Schedule     |     2000000 | 2021-08-17 10:09:45.628170900 | 2022-06-14 20:56:40.699043200 | 0            | 302              | Cobertura herdada do timestamp da corrida associada.      |
| rideestimative | Updated      |     2000000 | 2021-08-17 10:09:53.467805100 | 2022-06-14 20:56:55.425096800 | 0            | 302              | Cobertura herdada do timestamp da corrida associada.      |
| product        | N/A          |          34 | NaT                           | NaT                           | <NA>         | <NA>             | Tabela sem campos temporais nativos no dataset analitico. |

## Gaps Temporais

| gap_start   | gap_end   |   gap_days | note                                               |
|:------------|:----------|-----------:|:---------------------------------------------------|
| NaT         | NaT       |          0 | Nenhum gap temporal em Create no nivel de corrida. |

## Integridade dos Timestamps

| check_name                   |   count |   pct_of_rides | status                | note                                                                                                |
|:-----------------------------|--------:|---------------:|:----------------------|:----------------------------------------------------------------------------------------------------|
| schedule_before_create_any   |  238854 |        99.8261 | expected_micro_offset | Schedule tende a anteceder Create por milissegundos em corridas on-demand.                          |
| schedule_create_abs_gt_1_min |     159 |         0.0665 | review                | Gap maior que 1 minuto entre Schedule e Create; vale inspecionar, embora nao haja casos >5 minutos. |
| schedule_create_abs_gt_5_min |       0 |         0      | ok                    | Nao foram encontrados casos acima de 5 minutos.                                                     |
| updated_before_create        |       0 |         0      | ok                    | Nao ha timestamps de Updated anteriores a Create.                                                   |
| updated_before_schedule      |       0 |         0      | ok                    | Nao ha timestamps de Updated anteriores a Schedule.                                                 |
| updated_after_1_hour         |     172 |         0.0719 | review                | Atualizacoes tardias sao raras e sugerem eventos operacionais fora da janela imediata.              |
| updated_after_24_hours       |      59 |         0.0247 | review                | Atualizacoes acima de 24h nao devem ser usadas como ancora temporal para TSCV.                      |
| updated_after_7_days         |      30 |         0.0125 | review                | Casos muito tardios reforcam que Updated e inadequado como indice temporal principal.               |

## Sazonalidade Diaria

| date                |   unique_rides |   estimative_rows |   mean_price |   median_price |   p95_price |
|:--------------------|---------------:|------------------:|-------------:|---------------:|------------:|
| 2021-08-17 00:00:00 |           2499 |             20117 |      34.3298 |          22.68 |      90     |
| 2021-08-18 00:00:00 |           3266 |             25683 |      38.0174 |          25.73 |     107.826 |
| 2021-08-19 00:00:00 |           3258 |             25599 |      32.7895 |          23.27 |      88     |
| 2021-08-20 00:00:00 |           3493 |             27192 |      38.2052 |          26    |     111.444 |
| 2021-08-21 00:00:00 |           2101 |             16441 |      38.061  |          27.5  |     101     |
| 2021-08-22 00:00:00 |             46 |               352 |     121.973  |          27.12 |     124.67  |
| 2021-08-23 00:00:00 |           3938 |             29806 |      37.2731 |          26.5  |     102     |
| 2021-08-24 00:00:00 |           3327 |             26255 |      34.2845 |          23.5  |      89     |
| 2021-08-25 00:00:00 |           3292 |             26429 |      37.0915 |          25    |      95.18  |
| 2021-08-26 00:00:00 |           3125 |             24738 |      37.7963 |          24.5  |      89.5   |

## Sazonalidade Semanal

| weekday   |   avg_unique_rides |   median_unique_rides |   avg_estimative_rows |   avg_mean_price |   median_mean_price |   observed_days |
|:----------|-------------------:|----------------------:|----------------------:|-----------------:|--------------------:|----------------:|
| Monday    |           926.674  |                   281 |              7585.19  |          41.9547 |             41.1971 |              43 |
| Tuesday   |           877.136  |                   324 |              7514.11  |          44.3652 |             44.3877 |              44 |
| Wednesday |          1038.95   |                   345 |              8771.58  |          43.5436 |             39.8752 |              43 |
| Thursday  |          1046.72   |                   349 |              8902.77  |          43.727  |             43.941  |              43 |
| Friday    |          1035.7    |                   280 |              8529.12  |          44.8095 |             45.1288 |              43 |
| Saturday  |           570.535  |                    91 |              4603.86  |          41.6907 |             39.7281 |              43 |
| Sunday    |            48.3023 |                    52 |               430.256 |          50.6976 |             50.8267 |              43 |

## Sazonalidade Mensal

| month   |   active_days |   total_unique_rides |   avg_unique_rides |   avg_estimative_rows |   avg_mean_price |   median_mean_price |
|:--------|--------------:|---------------------:|-------------------:|----------------------:|-----------------:|--------------------:|
| 2021-08 |            15 |                40944 |          2729.6    |             21382.6   |          42.4711 |             37.2731 |
| 2021-09 |            30 |                76603 |          2553.43   |             19860.6   |          41.7102 |             38.0575 |
| 2021-10 |            31 |                75669 |          2440.94   |             20008.9   |          37.4504 |             37.2865 |
| 2021-11 |            30 |                 8925 |           297.5    |              2425.67  |          41.9079 |             42.1412 |
| 2021-12 |            31 |                 2977 |            96.0323 |               792.226 |          54.3337 |             48.3973 |
| 2022-01 |            31 |                 2578 |            83.1613 |               776.935 |          38.2721 |             35.6456 |
| 2022-02 |            28 |                 4350 |           155.357  |              1606.29  |          41.6897 |             38.932  |
| 2022-03 |            31 |                 6267 |           202.161  |              2116.52  |          45.4874 |             44.1316 |
| 2022-04 |            30 |                 7455 |           248.5    |              2677.67  |          49.6511 |             46.8954 |
| 2022-05 |            31 |                 9591 |           309.387  |              3501.55  |          47.5171 |             46.6171 |
| 2022-06 |            14 |                 3911 |           279.357  |              3020.5   |          49.3517 |             50.1148 |

## Interpretacao DS

- A serie e continua de `2021-08-17` a `2022-06-14`, sem gaps diarios em `Create` no nivel de corrida.
- `Schedule` antecede `Create` em `99%+` dos casos por milissegundos; apenas `159` corridas passam de 1 minuto, e nenhuma passa de 5 minutos.
- `Updated` nunca antecede `Create`, mas possui `59` atualizacoes acima de 24h; por isso, nao deve ser usado como eixo do TSCV.
- Existe sazonalidade semanal forte no volume: quinta-feira tem em media `~1047` corridas/dia, enquanto domingo cai para `~48`.
- O preco medio e menos estavel que o volume, mas mostra patamar mais alto a partir do regime de baixo volume, especialmente de `abr/2022` em diante.
- Ha quebra de regime clara no volume a partir de `nov/2021`, com queda muito forte frente a `ago-out/2021`. Isso precisa entrar na definicao dos folds.

## Recomendacao para TSCV

- Use `Create` como ancora temporal principal do TSCV. `Schedule` e praticamente identico a `Create` neste dataset, enquanto `Updated` possui cauda longa de atualizacao tardia.
- Como existe sazonalidade semanal forte, cada janela de validacao deve cobrir semanas completas. O minimo recomendado e `28 dias` por fold.
- O dataset cobre `302` dias continuos, o que comporta `5 folds` de `28 dias` com janela expansiva no historico completo. Se a prioridade for estabilidade de regime, use o subperiodo `2021-11-01` a `2022-06-14` (226 dias) e rode `4 folds` de `28 dias`.
- Ha mudanca forte de regime no volume: media diaria de `~2542` corridas entre `ago-out/2021` contra `~204` entre `nov/2021-jun/2022`. Por isso, as metricas mais representativas do futuro recente tendem a vir dos folds posicionados no regime mais novo.
- Como ha varias estimativas por corrida, todos os registros do mesmo `RideID` devem permanecer no mesmo fold para evitar leakage entre treino e validacao.

## Artefatos

- Daily overview: `temporal_daily_overview.png`
- Weekly seasonality: `temporal_weekday_seasonality.png`
- Monthly metrics: `temporal_monthly_metrics.png`

## Conclusao

- A cobertura temporal e suficiente para TSCV, mas os folds precisam respeitar semanas completas e a mudanca de regime observada no volume.
- `Create` e o timestamp mais confiavel para ordenar os dados. `Updated` deve ficar fora da definicao dos splits.
- A avaliacao mais representativa do cenario recente tende a vir de validacoes concentradas no periodo `nov/2021-jun/2022`.