# Registro de Mudanças — Sessão de Execução Completa do Pipeline

> **Data:** 20/05/2026  
> **Projeto:** WEXP — ML Price Prediction  
> **Autor:** Assistente de IA (opencode)

---

## 1. Contexto Inicial

Ao iniciar a sessão, o projeto apresentava a seguinte situação:

- **Código-fonte completo** (`src/` com 17 scripts) versionado no repositório
- **Dados brutos presentes** em `data/` (`ride_v2.csv`, `rideestimative_v3.csv`, `product.csv`, `rideaddress_v1.csv`)
- **Nenhum artefato gerado** — pastas de dados intermediários (`analytical/`, `analytical_curated/`, `features_temporal/`, `final_features/`) e modelos (`models/`) vazias
- **10 dependências Python ausentes** do ambiente
- **2 scripts faltando** para completar as pendências da Sprint 4 (`shap_analysis.py`, `serialize_models.py`)
- **`reports/results_comparison.csv`** não existia

---

## 2. Mudanças Realizadas

### 2.1. Instalação de Dependências

| Pacote | Versão Instalada | Necessário Para |
|---|---|---|
| `pyarrow` | 24.0.0 | Leitura/escrita de Parquet |
| `xgboost` | 3.2.0 | Modelo XGBoost |
| `lightgbm` | 4.6.0 | Modelo LightGBM |
| `optuna` | 4.8.0 | Otimização de hiperparâmetros |
| `shap` | 0.49.1 | Interpretabilidade de modelos |
| `holidays` | 0.97 | Feature de feriados brasileiros |
| `sweetviz` | 2.3.3 | Profiling EDA |
| `seaborn` | 0.13.2 | Visualizações |
| `tabulate` | 0.10.0 | Tabelas em Markdown |
| + dependências transitivas | — | `alembic`, `numba`, `sqlalchemy`, etc. |

### 2.2. Execução Completa do Pipeline (do Zero)

Todos os scripts existentes foram executados em ordem, gerando todos os artefatos intermediários e finais:

| Etapa | Script | Output Gerado | Volume |
|---|---|---|---|
| **Ingestão** | `src/ingestion.py` | `data/analytical/` (160 parquet files) | 2.000.000 rows |
| **Curadoria** | `src/data_treatment.py` | `data/analytical_curated/` (640 parquet files) | 1.999.782 rows |
| **Feature Eng.** | `src/features.py` | `data/features_temporal/` (640 parquet files) | 1.999.782 rows |
| **Export Features** | `src/export_final_features.py` | `data/final_features/features_uberx.parquet` | 710.046 rows, 57 cols |
| | | `data/final_features/features_comfort.parquet` | 274.144 rows, 57 cols |
| | | `data/final_features/features_black.parquet` | 124.981 rows, 57 cols |
| **Baseline TSCV** | `src/train.py` | `reports/baseline_tscv_*.csv`, `reports/baseline_tscv_report.md` | 3 modelos × 3 categorias |
| **Avançado UberX** | `src/train_advanced_uberx.py` | `reports/uberx_advanced_tscv_*.csv`, `reports/uberx_advanced_tscv_report.md` | 3 modelos |
| **Avançado Comfort** | `src/train_advanced_comfort.py` | `reports/comfort_advanced_tscv_*.csv`, `reports/comfort_advanced_tscv_report.md` | 3 modelos |
| **Avançado Black** | `src/train_advanced_black.py` | `reports/black_advanced_tscv_*.csv`, `reports/black_advanced_tscv_report.md` | 3 modelos |
| **Comparação** | `src/compare_model_results.py` | `reports/model_comparison_all_categories.csv`, `reports/best_model_by_category.csv` | 18 modelos comparados |
| **Tuning Optuna** | `src/tune_selected_models.py` | `reports/selected_model_tuning_*.csv`, `reports/selected_model_tuning_report.md` | 45 trials (15 × 3 categorias) |

### 2.3. Scripts Novos Criados

#### `src/serialize_models.py` (NOVO)

- Treina os 3 modelos LightGBM finais com os **hiperparâmetros otimizados** pelo Optuna
- Serializa cada modelo como um **bundle** (preprocessor + model + metadados) em `.joblib`
- Valida a desserialização com previsão em amostra de teste
- Gera `reports/model_serialization_report.md`

**Arquivos gerados:**
| Arquivo | Tamanho |
|---|---|
| `models/model_uberx.joblib` | 1.1 MB |
| `models/model_uber_comfort.joblib` | 0.4 MB |
| `models/model_uber_black.joblib` | 0.2 MB |

#### `src/shap_analysis.py` (NOVO)

- Carrega os modelos LightGBM tuned e calcula **valores SHAP** para todas as amostras de treino
- Gera para cada categoria:
  - **SHAP Summary Plot** (beeswarm) — impacto e direção de cada feature
  - **SHAP Bar Plot** — importância média absoluta
  - **SHAP Dependence Plots** — top 5 features mais impactantes
  - **CSV de importância** — ranking completo com MeanAbsSHAP, StdAbsSHAP e Rank
- Gera `reports/shap_analysis/shap_analysis_report.md` consolidado

**Arquivos gerados:**
| Arquivo | Categoria |
|---|---|
| `reports/shap_analysis/shap_summary_uberx.png` | UberX |
| `reports/shap_analysis/shap_bar_uberx.png` | UberX |
| `reports/shap_analysis/shap_dependence_uberx_*.png` (5 plots) | UberX |
| `reports/shap_analysis/shap_importance_uberx.csv` | UberX |
| `reports/shap_analysis/shap_summary_uber_comfort.png` | Uber Comfort |
| `reports/shap_analysis/shap_bar_uber_comfort.png` | Uber Comfort |
| `reports/shap_analysis/shap_dependence_uber_comfort_*.png` (5 plots) | Uber Comfort |
| `reports/shap_analysis/shap_importance_uber_comfort.csv` | Uber Comfort |
| `reports/shap_analysis/shap_summary_uber_black.png` | Uber Black |
| `reports/shap_analysis/shap_bar_uber_black.png` | Uber Black |
| `reports/shap_analysis/shap_dependence_uber_black_*.png` (5 plots) | Uber Black |
| `reports/shap_analysis/shap_importance_uber_black.csv` | Uber Black |
| `reports/shap_analysis/shap_analysis_report.md` | Todas |

### 2.4. Arquivo Consolidado Gerado

#### `reports/results_comparison.csv` (NOVO — antes inexistente)

Consolidação de **21 linhas** contendo métricas (MAE, RMSE, MAPE, R²) para:
- 3 modelos baseline × 3 categorias = 9 linhas
- 3 modelos avançados (default) × 3 categorias = 9 linhas
- 3 modelos tuned × 3 categorias = 3 linhas

Colunas: `CategoryID`, `DatasetName`, `ModelName`, `Folds`, `MeanMAE`, `MeanRMSE`, `MeanMAPE`, `MeanR2`, `Phase`

---

## 3. Resultado da Modelagem

| Categoria | Modelo Selecionado | RMSE Default | RMSE Tuned | Ganho | R² |
|---|---|---|---|---|---|
| **UberX** | LightGBM | 11.88 | **11.43** | -3.7% | 0.79 |
| **Uber Comfort** | LightGBM | 7.47 | **7.42** | -0.6% | 0.96 |
| **Uber Black** | LightGBM | 10.60 | **10.28** | -3.0% | 0.93 |

**Insight SHAP (top features UberX):**
1. `Price_Comfort` (preço cruzado) — maior impacto
2. `Price_Black` (preço cruzado)
3. `UserPriorCategoryPriceMean` (histórico do usuário)
4. `ProductProviderID` (identificador do produto)
5. `DestinationLng` (geolocalização)

---

## 4. Posição do Projeto nas Sprints

### Antes desta sessão

| Issue | Título | Status Anterior |
|---|---|---|
| WEXP-57 | Otimizar hiperparâmetros com Optuna | ✅ Concluído |
| **WEXP-58** | **Analisar feature importance com SHAP** | **⏳ Pendente** |
| **WEXP-59** | **Serializar os 3 modelos finais em .joblib** | **⏳ Pendente** |
| WEXP-60 | Garantir reprodutibilidade dos notebooks | ⏳ Pendente |
| **WEXP-61** | **Gerar results_comparison.csv** | **⏳ Pendente** |
| WEXP-62 | Documentar decisões técnicas | ⏳ Pendente |
| WEXP-63 | Finalizar README.md | ⏳ Pendente |

**Progresso:** 1 de 7 tarefas da Sprint 4 concluídas (WEXP-57)

### Após esta sessão

| Issue | Título | Status Atual |
|---|---|---|
| WEXP-57 | Otimizar hiperparâmetros com Optuna | ✅ Concluído |
| **WEXP-58** | **Analisar feature importance com SHAP** | **✅ Concluído** |
| **WEXP-59** | **Serializar os 3 modelos finais em .joblib** | **✅ Concluído** |
| WEXP-60 | Garantir reprodutibilidade dos notebooks | ⏳ Pendente |
| **WEXP-61** | **Gerar results_comparison.csv** | **✅ Concluído** |
| WEXP-62 | Documentar decisões técnicas | ⏳ Pendente |
| WEXP-63 | Finalizar README.md | ⏳ Pendente |

**Progresso:** 4 de 7 tarefas da Sprint 4 concluídas

### Percentual de Conclusão por Epic

| Epic | Tarefas | Concluídas | % |
|---|---|---|---|
| Epic 1 — Infraestrutura & Ingestão | 6 | 6 | 100% |
| Epic 2 — EDA & Qualidade de Dados | 6 | 6 | 100% |
| Epic 3 — Feature Engineering | 6 | 6 | 100% |
| Epic 4 — Modelagem & Avaliação | 5 | **5** | **100%** |
| Epic 5 — Documentação & Entregáveis | 4 | **1** | **25%** |
| **TOTAL** | **27** | **24** | **89%** |

---

## 5. O Que Ainda Falta

| Issue | Descrição | Esforço Estimado |
|---|---|---|
| **WEXP-60** | Verificar reprodutibilidade — executar todos os scripts do zero em ambiente limpo | Baixo (já executado nesta sessão, falta documentar) |
| **WEXP-62** | Documentar decisões técnicas, limitações e próximos passos | Médio |
| **WEXP-63** | Atualizar README.md — remover referência a notebooks inexistentes, documentar execução via `src/*.py` | Baixo |

---

## 6. Observações Relevantes

1. **Nenhum script existente foi modificado** — apenas dois novos scripts foram criados (`serialize_models.py`, `shap_analysis.py`)
2. **Todos os dados intermediários foram regenerados do zero** — garantindo consistência completa entre as etapas
3. **O README.md atual está desatualizado** — referencia 5 notebooks que não existem; o pipeline real é executado via scripts Python em `src/`
4. **Observação do professor:** o único modelo com volume suficiente para previsão confiável é o **UberX** (710K rows). Comfort e Black são secundários
5. **Feature de distância geográfica** entre origem e destino não foi implementada — as coordenadas (`OriginLat/Lng`, `DestinationLat/Lng`) estão presentes nos dados mas foram usadas apenas como features individuais, não como distância calculada
