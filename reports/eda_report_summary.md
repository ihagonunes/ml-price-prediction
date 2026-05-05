# EDA Consolidado das Categorias-Alvo

- Gerado em: `2026-05-05T15:45:48.752968`
- Fonte: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical`
- Relatorio HTML: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\reports\eda_report.html`
- Populacao analisada (categorias 2, 9, 4): `1109259` linhas
- Amostra Sweetviz: `300000` linhas
- Cobertura temporal: `2021-08-17 10:09:45.642582500` ate `2022-06-14 20:56:40.757298900`

## Cobertura por Categoria

| CategoryName   |   population_rows |   sample_rows |
|:---------------|------------------:|--------------:|
| UberX          |            710134 |        100000 |
| Uber Comfort   |            274144 |        100000 |
| Uber Black     |            124981 |        100000 |

## Principais Achados DS

- `UberX` concentra o maior volume (`710134` linhas) e continua com a cauda de `Price` mais longa entre as categorias-alvo.
- `Uber Black` tem o maior nivel central de preco (mediana `44.0` e p99 `177.5`), mesmo com menor volume.
- `Uber Comfort` fica no meio do caminho em volume e patamar de preco, mas mantem assimetria forte (`skew=55.5139`).
- O missing de `FareID` permanece alto nas tres categorias (`UberX=50.03%`, `Uber Comfort=2.88%`, `Uber Black=1.04%`).
- `WaitingTime` segue concentrado em janelas curtas; o p95 por categoria ficou em `UberX=10.0`, `Uber Comfort=10.0` e `Uber Black=14.0` minutos.
- O volume semanal segue concentrado em dias uteis; os tres maiores dias no recorte foram `{'Thursday': 211132, 'Wednesday': 207939, 'Friday': 204772}`.

## Observacao de Configuracao

- O HTML foi gerado com `sweetviz`, `pairwise_analysis='off'` e amostragem estratificada deterministica de ate `100000` linhas por categoria para manter o processo reproduzivel e o artefato navegavel no volume atual.