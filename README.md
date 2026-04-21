# Modelo Preditivo de Preços por Categoria de Serviço

**Área:** Martech / Analytics  
**Time:** Data Science & Engenharia de Dados  
**Status:** Em desenvolvimento  

---

## Objetivo

Desenvolver três modelos de Machine Learning independentes (um por categoria) para prever o preço estimado de corridas nas categorias **UberX**, **Uber Comfort** e **Uber Black**, utilizando como entrada as características da corrida presentes nas tabelas `ride`, `rideestimative` e `product`.

A variável-alvo é o campo `Price` da tabela `rideestimative`. O campo `price` da tabela `ride` (preço real) **não pode ser utilizado como feature**.

---

## Estrutura do Repositório

```
ml-price-prediction/
│
├── data/                        # Dados brutos (não versionados — ver .gitignore)
│   └── README.md                # Instruções de como obter os dados
│
├── notebooks/                   # Notebooks de análise e modelagem
│   ├── 01_eda.ipynb             # Análise exploratória de dados
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_uberx.ipynb
│   ├── 03_modeling_comfort.ipynb
│   └── 03_modeling_black.ipynb
│
├── src/                         # Código-fonte modularizado
│   ├── ingestion.py             # Ingestão, validação e join das tabelas
│   ├── features.py              # Feature engineering
│   ├── train.py                 # Loop de TSCV, treino e avaliação
│   └── utils.py                 # Helpers compartilhados
│
├── models/                      # Modelos serializados (não versionados)
│   └── .gitkeep
│
├── reports/                     # Relatórios gerados
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dados

Os arquivos de dados **não estão versionados** no repositório por questões de tamanho e confidencialidade. Para reproduzir o projeto, coloque os seguintes arquivos na pasta `data/`:

| Arquivo | Descrição | Separador | Encoding |
|---|---|---|---|
| `ride.csv` | Corridas (~1.6M registros) | `;` | UTF-8 |
| `rideestimative.csv` | Estimativas de preço (~2.5 GB) | `;` | UTF-8 |
| `product.csv` | Catálogo de produtos/categorias (237 registros) | `;` | UTF-8 |

> ⚠️ O arquivo `rideestimative.csv` tem ~2.5 GB. A leitura é feita com `chunksize` para evitar estouro de memória.

---

## Como Reproduzir

### 1. Clonar o repositório

```bash
git clone https://github.com/<org>/ml-price-prediction.git
cd ml-price-prediction
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Adicionar os dados

Coloque os três arquivos CSV na pasta `data/` conforme a tabela acima.

### 5. Executar os notebooks em ordem

| Ordem | Notebook | Descrição |
|---|---|---|
| 1 | `01_eda.ipynb` | EDA e profiling das 3 tabelas |
| 2 | `02_feature_engineering.ipynb` | Geração dos datasets de features por categoria |
| 3 | `03_modeling_uberx.ipynb` | Modelagem para UberX |
| 4 | `03_modeling_comfort.ipynb` | Modelagem para Uber Comfort |
| 5 | `03_modeling_black.ipynb` | Modelagem para Uber Black |

> Cada notebook deve ser executado com **Restart Kernel & Run All** para garantir reprodutibilidade.

---

## Entregáveis

| Artefato | Localização |
|---|---|
| 3 modelos serializados | `models/model_uberx.joblib`, `model_comfort.joblib`, `model_black.joblib` |
| Relatório de EDA | `reports/eda_report.html` |
| Comparativo de métricas | `reports/results_comparison.csv` |

---

## Estratégia de Validação

Os modelos são avaliados com **Time Series Cross-Validation (TSCV)** — folds ordenados por tempo, sem shuffling. O modelo nunca "vê o futuro" durante treino ou validação.

**Métricas reportadas por fold e média geral:** MAE, RMSE, MAPE, R²

---

## Regras Críticas

- ❌ O campo `price` da tabela `ride` **nunca** deve ser usado como feature
- ❌ Campos PII (`Name`, `Phone`, `Driver`, `Plate`, `DriverPhone`, `DriverPicture`, `Registration`) são removidos automaticamente no pipeline de ingestão
- ✅ A variável-alvo é exclusivamente `Price` da tabela `rideestimative`

---

## Categorias-Alvo

| Categoria | ProviderID | CategoryID | Produtos |
|---|---|---|---|
| UberX | 2 | 2 | UberX, UberXPromo, UberX Sem Pressa |
| Uber Comfort | 2 | 9 | Comfort, Select, VoucherComfort |
| Uber Black | 2 | 4 | Black, WPP5, WPP-5-5 |

---

## Fora do Escopo

- Deploy em produção (API de inferência, monitoramento, retraining automático)
- Integração com data lakes corporativos
- Modelos para categorias além das três especificadas
