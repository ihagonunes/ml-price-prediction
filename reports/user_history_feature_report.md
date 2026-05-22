# Features Historicas por Usuario

- Gerado em: `2026-05-20T16:36:01.204478`
- Fonte principal: `C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\data\analytical_curated`
- Fonte de preco pago historico: `C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\data\analytical`
- Saida com features: `C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\data\features_temporal`

## Features Criadas

- `UserPriorRideCount`: quantidade de corridas anteriores do usuario no nivel de `RideID`.
- `UserPriorPaidPriceMean`: media expansiva do `RidePrice` das corridas anteriores do usuario, excluindo o registro atual.
- `UserPriorCategoryRideCount`: quantidade de ocorrencias anteriores do usuario na mesma `CategoryID`, calculada no nivel canonico de `RideID + CategoryID`.
- `UserPriorCategoryPriceMean`: ticket medio historico do usuario para a mesma `CategoryID`, usando o `Price` canonico anterior e excluindo a corrida atual.

## Regra Anti-Leakage

- A ancora temporal e `Create`, que ja foi validada como o melhor indice temporal do projeto.
- As features gerais do usuario sao calculadas no nivel de `RideID`, evitando que varias estimativas da mesma corrida pesem mais de uma vez.
- As features por categoria usam uma versao canonica de `RideID + CategoryID`, deduplicada pelo menor `RideEstimativeID`, para evitar leakage entre ofertas repetidas da mesma corrida.
- Em ambos os casos, a janela e expansiva e exclui o registro corrente via `cumcount` e `cumsum` com deslocamento logico.

## Decisao DS

- `UserPriorRideCount` captura maturidade e frequencia de uso do cliente.
- `UserPriorPaidPriceMean` aproxima o ticket historico real do usuario e ajuda a separar perfis sensiveis a preco de perfis premium.
- `UserPriorCategoryRideCount` e `UserPriorCategoryPriceMean` trazem preferencia e faixa de preco historica por categoria, sinal especialmente util para `UberX`, `Comfort` e `Black`.

## Cobertura por Categoria-Alvo

|   CategoryID | CategoryName   | FeatureName                | FeatureRole    |   available_rows |   available_pct |
|-------------:|:---------------|:---------------------------|:---------------|-----------------:|----------------:|
|            2 | UberX          | UserPriorCategoryRideCount | category_count |           700404 |         98.6421 |
|            2 | UberX          | UserPriorCategoryPriceMean | category_value |           700404 |         98.6421 |
|            2 | UberX          | UserPriorRideCount         | history_count  |           700406 |         98.6423 |
|            2 | UberX          | UserPriorPaidPriceMean     | history_value  |           700331 |         98.6318 |
|            4 | Uber Black     | UserPriorCategoryRideCount | category_count |           122301 |         97.8557 |
|            4 | Uber Black     | UserPriorCategoryPriceMean | category_value |           122301 |         97.8557 |
|            4 | Uber Black     | UserPriorRideCount         | history_count  |           122738 |         98.2053 |
|            4 | Uber Black     | UserPriorPaidPriceMean     | history_value  |           122699 |         98.1741 |
|            9 | Uber Comfort   | UserPriorCategoryRideCount | category_count |           269164 |         98.1834 |
|            9 | Uber Comfort   | UserPriorCategoryPriceMean | category_value |           269164 |         98.1834 |
|            9 | Uber Comfort   | UserPriorRideCount         | history_count  |           269565 |         98.3297 |
|            9 | Uber Comfort   | UserPriorPaidPriceMean     | history_value  |           269483 |         98.2998 |

## Integridade

- Linhas antes/depois do pipeline: `1999782` / `1999782`.
- Duplicatas colapsadas no historico por corrida: `1760577`.
- Duplicatas colapsadas no historico por `RideID + CategoryID`: `894314`.
- Cold start no nivel de corrida: `3699` corridas unicas.
- Cold start no nivel de categoria: `21530` pares canonicos `RideID + CategoryID`.