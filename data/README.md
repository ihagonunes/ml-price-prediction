# Dados

Os arquivos de dados **não são versionados** neste repositório.

## Arquivos necessários

Coloque os seguintes arquivos nesta pasta antes de executar o pipeline:

| Arquivo | Separador | Encoding | Observação |
|---|---|---|---|
| `ride_v2.csv` | `;` (ponto-e-vírgula) | UTF-8 | ~300 MB |
| `rideestimative_v3.csv` | `;` (ponto-e-vírgula) | UTF-8 | ~2.5 GB |
| `rideaddress_v1.csv` | `;` (ponto-e-vírgula) | UTF-8 | — |
| `product.csv` | `;` (ponto-e-vírgula) | UTF-8 | < 1 MB |

## Atenção

- O arquivo `rideestimative_v3.csv` é grande (~2.5 GB). Não tente abri-lo diretamente no Excel ou em editores de texto.
- A leitura é feita via `pandas` com `chunksize=100_000` para evitar estouro de memória.
- Nomes legados (`ride.csv`, `rideestimative.csv`) não são reconhecidos pelo pipeline: renomeie ou ajuste as constantes em `src/ingestion.py`.
- Os arquivos de dados originais são distribuídos fora deste repositório (pacote do projeto).