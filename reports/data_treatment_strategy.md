# Estrategias de Tratamento de Nulos, Outliers e Inconsistencias

- Gerado em: `2026-05-05T15:26:04.800675`
- Fonte: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical`
- Saida curada: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical_curated`
- Linhas na origem: `2000000`
- Linhas apos tratamento: `1999782`
- Linhas removidas: `218`
- Retencao: `99.9891%`

## Regras Definidas

| field_scope                                                                          | issue                                                      | strategy                                                      | implementation                                                                                                                                                    | reason                                                                                                                   |
|:-------------------------------------------------------------------------------------|:-----------------------------------------------------------|:--------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| FareID                                                                               | 40.91% de nulos em rideestimative.                         | Imputar com sentinel categorico.                              | Preencher nulos com MISSING_FAREID e criar a flag FareIDWasImputed.                                                                                               | FareID e um identificador categorico; a imputacao explicita preserva missingness como sinal sem distorcer a moda.        |
| WaitingTime, Fee, TotalUsers                                                         | Campos numericos retidos podem receber nulos futuros.      | Fallback de mediana.                                          | Aplicar mediana global apenas se surgirem nulos; nesta execucao nao houve imputacoes.                                                                             | A mediana e robusta e nao puxa a distribuicao por caudas extremas.                                                       |
| Price                                                                                | Assimetria forte e outliers extremos por categoria.        | Remover Price <= 0 e aplicar capping superior por CategoryID. | Remover registros com Price <= 0 e capar o restante no percentil 99.5 por CategoryID.                                                                             | O target tem comportamento diferente por categoria; o capping segmentado preserva volume e reduz distorcao nos extremos. |
| WaitingTime                                                                          | Cauda longa com outliers detectados no profiling inicial.  | Capping superior global.                                      | Capar no percentil 99 global e registrar a flag WaitingTimeWasCapped.                                                                                             | A feature e operacional e monotona; o clipping leve reduz extremos sem descartar observacoes.                            |
| RidePrice, Selected, RideReasonSelectedEstimativeID                                  | Vazamento e informacao pos-evento.                         | Remocao de colunas.                                           | Excluir do dataset curado.                                                                                                                                        | Nao estao disponiveis de forma honesta no momento de inferencia.                                                         |
| Car, ProviderID, RideProviderID, RideCategoryID, RideDriverLocationID, ScheduledRide | Esparsidade extrema ou coluna constante.                   | Remocao de colunas.                                           | Excluir do dataset curado.                                                                                                                                        | Mais de 99% de nulos ou ausencia de variancia tornam essas colunas custosas e pouco informativas.                        |
| RideEstimativeID                                                                     | Risco de duplicidade de chave de negocio.                  | Remocao de duplicatas por chave primaria.                     | Manter a primeira ocorrencia de cada RideEstimativeID.                                                                                                            | Garante 1 linha por estimativa e evita inflar treino/validacao.                                                          |
| Create, Schedule, Updated                                                            | Necessidade de integridade temporal para TSCV e auditoria. | Padronizacao datetime e remocao de cronologias impossiveis.   | Converter para datetime, remover registros com Create/Schedule invalidos, Updated < Create ou gap absoluto > 5 min entre Schedule e Create em corridas on-demand. | Create e a ancora temporal confiavel; cronologias impossiveis devem sair do dataset.                                     |
| OriginLat, OriginLng, DestinationLat, DestinationLng                                 | Campos espaciais obrigatorios para FE geografica.          | Remocao de linhas invalidas.                                  | Descartar linhas com coordenadas ausentes ou nao numericas.                                                                                                       | Distancia e zona geografica dependem de coordenadas validas.                                                             |

## Thresholds de Capping

| feature     | group_key   |   quantile |   upper_cap |
|:------------|:------------|-----------:|------------:|
| WaitingTime | ALL         |      0.99  |      15     |
| Price       | 1           |      0.995 |     192.123 |
| Price       | 2           |      0.995 |     163.407 |
| Price       | 4           |      0.995 |     198.5   |
| Price       | 5           |      0.995 |     251.63  |
| Price       | 6           |      0.995 |     312.432 |
| Price       | 8           |      0.995 |     267.4   |
| Price       | 9           |      0.995 |     210     |
| Price       | 10          |      0.995 |     125.15  |

## Impacto da Execucao

| metric                                  |            value |
|:----------------------------------------|-----------------:|
| source_rows                             |      2e+06       |
| rows_written                            |      1.99978e+06 |
| rows_removed_total                      |    218           |
| retention_pct                           |     99.9891      |
| duplicate_rideestimative_rows_removed   |      0           |
| exact_duplicate_rows_removed            |      0           |
| invalid_create_or_schedule_rows_removed |      0           |
| updated_before_create_rows_removed      |      0           |
| schedule_create_gap_gt_5m_rows_removed  |      0           |
| missing_coordinate_rows_removed         |    215           |
| nonpositive_price_rows_removed          |      3           |
| fareid_sentinel_imputations             | 818007           |
| waitingtime_median_imputations          |      0           |
| fee_median_imputations                  |      0           |
| totalusers_median_imputations           |      0           |
| waiting_time_rows_capped                |  12591           |
| price_rows_capped                       |   9981           |

## Distribuicao das Particoes Curadas

|   CategoryID |   rows_written |   price_rows_capped |
|-------------:|---------------:|--------------------:|
|            1 |          51352 |                 257 |
|            2 |         710046 |                3551 |
|            4 |         124981 |                 617 |
|            5 |         439155 |                2194 |
|            6 |          28122 |                 141 |
|            8 |          36641 |                 184 |
|            9 |         274144 |                1360 |
|           10 |         335341 |                1677 |

## Interpretacao DS

- `FareID` ficou com imputacao categorica explicita porque o missing e volumoso e potencialmente informativo; forcar a moda esconderia esse padrao.
- `Price` passou a usar capping superior no percentil 99.5 por `CategoryID`, o que respeita a heterogeneidade entre produtos e reduz a influencia dos extremos.
- `WaitingTime` recebeu capping leve no percentil 99 global para reduzir cauda operacional sem descarte de volume.
- Colunas com vazamento (`RidePrice`, `Selected`, `RideReasonSelectedEstimativeID`) e colunas ultra-esparsas/constantes foram removidas do dataset curado.
- Registros temporalmente impossiveis, coordenadas invalidas e `Price <= 0` foram configurados para remocao por regra de negocio.

## Conclusao

- O pipeline agora produz uma camada `analytical_curated` pronta para EDA orientada a modelagem e para a proxima etapa de feature engineering.
- O dataset bruto consolidado em `data/analytical` permanece preservado para auditoria e reproducibilidade.