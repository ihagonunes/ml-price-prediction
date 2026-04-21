# Dados

Os arquivos de dados não são versionados neste repositório.

## Arquivos necessários

Coloque os seguintes arquivos nesta pasta antes de executar os notebooks:

| Arquivo | Tamanho aproximado | Separador | Encoding |
|---|---|---|---|
| `ride.csv` | ~300 MB | `;` (ponto-e-vírgula) | UTF-8 |
| `rideestimative.csv` | ~2.5 GB | `;` (ponto-e-vírgula) | UTF-8 |
| `product.csv` | < 1 MB | `;` (ponto-e-vírgula) | UTF-8 |

## Atenção

- O arquivo `rideestimative.csv` é grande (~2.5 GB). Não tente abri-lo diretamente no Excel ou em editores de texto.
- A leitura é feita via `pandas` com `chunksize=100_000` para evitar estouro de memória.
- Os arquivos estão disponíveis no arquivo `.rar` original do projeto.
