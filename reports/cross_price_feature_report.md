# Features Cruzadas de Preco entre Categorias

- Gerado em: `2026-05-20T16:36:01.201329`
- Fonte: `C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\data\analytical_curated`
- Saida com features: `C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\data\features_temporal`

## Regra de Pivot

- O pivot usa apenas os produtos canonicos `UberX`, `Comfort` e `Black` para gerar `Price_UberX`, `Price_Comfort` e `Price_Black`.
- A chave do lookup e `RideID`, com deduplicacao rara em `RideID + ProductID` pelo menor `RideEstimativeID`, para privilegiar a primeira estimativa disponivel e evitar usar refreshes posteriores.
- O merge de volta na base principal e `many-to-one` por `RideID`, sem aumento do numero de linhas.

## Regra DS de Uso sem Leakage

- No modelo `UberX`, usar `Price_Comfort` e `Price_Black` como auxiliares e excluir `Price_UberX` da matriz de features.
- No modelo `Comfort`, usar `Price_UberX` e `Price_Black` como auxiliares e excluir `Price_Comfort`.
- No modelo `Black`, usar `Price_UberX` e `Price_Comfort` como auxiliares e excluir `Price_Black`.
- Como o schema nao traz timestamp proprio por estimativa, `RideEstimativeID` foi usado como melhor proxy de ordem dentro da mesma corrida.

## Cobertura das Features Cruzadas

| ModelProduct   | FeatureName           | FeatureRole       |   available_rows |   available_pct |
|:---------------|:----------------------|:------------------|-----------------:|----------------:|
| Black          | Price_Comfort         | auxiliary_input   |           118450 |         95.7737 |
| Black          | Price_UberX           | auxiliary_input   |           123604 |         99.941  |
| Black          | Both auxiliary prices | auxiliary_pair    |           118418 |         95.7478 |
| Black          | Price_Black           | target_equivalent |           123677 |        100      |
| Comfort        | Price_Black           | auxiliary_input   |           118450 |         61.3788 |
| Comfort        | Price_UberX           | auxiliary_input   |           192794 |         99.9026 |
| Comfort        | Both auxiliary prices | auxiliary_pair    |           118418 |         61.3622 |
| Comfort        | Price_Comfort         | target_equivalent |           192982 |        100      |
| UberX          | Price_Black           | auxiliary_input   |           123613 |         52.4375 |
| UberX          | Price_Comfort         | auxiliary_input   |           192810 |         81.7913 |
| UberX          | Both auxiliary prices | auxiliary_pair    |           118427 |         50.2376 |
| UberX          | Price_UberX           | target_equivalent |           235734 |        100      |

## Integridade

- Duplicatas canonicas colapsadas no pivot: `326`.
- Corridas com as tres estimativas canonicas disponiveis: `118333`.
- Linhas antes/depois do join: `1999782` / `1999782`.