# Modelo Preditivo de Preços por Categoria de Serviço

**Área:** Martech / Analytics

**Time:** Data Science & Engenharia de Dados

**Status:** Concluído

## Objetivo

Prever o preço estimado do `Price` das corridas em três categorias independentes:

- `UberX`
- `Uber Comfort`
- `Uber Black`

O projeto usa as tabelas `ride_v2.csv`, `rideestimative_v3.csv`, `rideaddress_v1.csv` e `product.csv` como fonte, com pipeline temporalmente ordenado, engenharia de features, validação por `TimeSeriesSplit` e tuning de hiperparâmetros com Optuna.

O campo **alvo** é `Price` da tabela `rideestimative_v3.csv`. O campo `price` da tabela `ride_v2.csv` é o preço real pago pelo cliente e **não pode ser usado como feature** (vazamento de informação).

## Stack

| Tecnologia | Uso |
|---|---|
| Python 3 (execução validada com Python 3.14) | Linguagem |
| pandas, numpy, pyarrow (Parquet) | Dados |
| scikit-learn | Baseline e métricas |
| LightGBM, XGBoost | Modelos avançados |
| Optuna | Tuning de hiperparâmetros |
| SHAP | Interpretabilidade |
| Sweetviz, matplotlib, seaborn | Profiling e EDA |
| holidays | Feriados brasileiros |
| Jupyter | Notebooks de apoio |

As versões estão fixadas em [`requirements.txt`](requirements.txt).

## Estrutura do Repositório

```text
ml-price-prediction/
  data/
    README.md            # Layout esperado dos dados (não versionados)
  docs/
    decisoes_tecnicas.md
  notebooks/
    eda_report_notes.ipynb
  src/                   # Pipeline de dados, modelagem e avaliação
  models/                # Modelos serializados (.joblib) — gerados, não versionados
  reports/               # Relatórios (MD/HTML/CSV/PNG) e resumos JSON (não versionados)
  tests/
    test_tscv_temporal_order.py
  requirements.txt
  README.md
```

## Como Obter os Dados

Os dados **não são versionados** neste repositório.

1. Extraia o pacote original do projeto e copie os arquivos para `data/` com os nomes esperados pelo pipeline:
   - `ride_v2.csv`
   - `rideestimative_v3.csv`
   - `rideaddress_v1.csv`
   - `product.csv`
2. Se tiver os arquivos com nomes legados (`ride.csv`, `rideestimative.csv`), renomeie-os ou ajuste as constantes em `src/ingestion.py`.

O arquivo `rideestimative_v3.csv` é grande e não deve ser aberto no Excel. A leitura do pipeline usa `chunksize` para evitar estouro de memória.

Veja também `data/README.md` para o layout esperado.

## Configuração do Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reprodução Completa Do Zero

Execute os passos na ordem abaixo para recriar o projeto do zero:

1. `python src/ingestion.py`
2. `python src/profiling.py`
3. `python src/target_analysis.py`
4. `python src/correlation_analysis.py`
5. `python src/temporal_analysis.py`
6. `python src/data_treatment.py`
7. `python src/eda_report.py`
8. `python src/features.py`
9. `python src/export_final_features.py`
10. `python src/train.py`
11. `python src/train_advanced_uberx.py`
12. `python src/train_advanced_comfort.py`
13. `python src/train_advanced_black.py`
14. `python src/compare_model_results.py`
15. `python src/tune_selected_models.py`
16. `python src/serialize_final_models.py`
17. `python src/generate_results_comparison.py`
18. `python src/validate_notebooks.py`

Essa ordem produz toda a cadeia de artefatos: dados curados, features finais, modelos baseline, modelos avançados, tuning, serialização final e consolidação comparativa.

## Validação Temporal

- Validação por `TimeSeriesSplit` com **4 folds**, mantendo a ordem cronológica dos dados (nov/2021 a mai/2022).
- Registros do mesmo `RideID` permanecem no mesmo fold.
- `Create` é a âncora temporal do treinamento (não usar `Updated`).

Detalhes completos em `reports/tscv_strategy.md`.

## Resultados

O melhor modelo final por **RMSE médio no TSCV** foi `LightGBM_Tuned` nas três categorias. As métricas abaixo são as médias dos folds (arquivo `reports/results_comparison.csv`):

| Categoria | Modelo final | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| UberX | LightGBM_Tuned | 5,93 | 11,43 | 17,60% | 0,808 |
| Uber Black | LightGBM_Tuned | 6,41 | 10,34 | 12,08% | 0,930 |
| Uber Comfort | LightGBM_Tuned | 4,20 | 7,42 | 8,69% | 0,958 |

Para referência, os baselines (Lasso, Regressão Linear e Ridge) ficaram atrás do modelo tuningado — por exemplo, no UberX o R² médio do baseline é ~0,67 com MAPE ~25%. O tuning usou `num_boost_round` 350–400 e 12–13 trials por categoria (Optuna). As datasets finais de treino usam 35 features (UberX), 30 (Uber Black) e 31 (Uber Comfort).

## Notebook De Apoio

- `notebooks/eda_report_notes.ipynb` — anotações DT ligadas ao `eda_report.html`. Para manter a reprodução limpa, use `Restart Kernel & Run All` ao reexecutá-lo, mesmo que seja atualmente apenas informativo.

## Testes

```bash
python -m pytest tests -q
```

A suíte valida a ordem temporal do `TimeSeriesSplit` (impede vazamento entre treino e teste). A reprodução dos notebooks é validada separadamente por `src/validate_notebooks.py`.

## Onde Encontrar Os Artefatos

### Dados intermediários (gerados, não versionados)

- `data/analytical/` — dataset analítico consolidado em Parquet;
- `data/analytical_curated/` — camada curada com tratamento de nulos, outliers e inconsistências;
- `data/features_temporal/` — features temporais e históricas por categoria;
- `data/final_features/` — datasets finais por categoria prontos para treino.

### Modelos (gerados, não versionados)

- `models/model_uberx.joblib`
- `models/model_comfort.joblib`
- `models/model_black.joblib`

### Relatórios (MD/HTML/CSV/PNG versionados)

- `reports/analytical_dataset_validation.json` (gerado)
- `reports/parquet_profiling_report.md`
- `reports/data_treatment_strategy.md`
- `reports/temporal_analysis.md`
- `reports/tscv_strategy.md`
- `reports/eda_report.html`
- `reports/baseline_tscv_report.md`
- `reports/uberx_advanced_tscv_report.md`
- `reports/comfort_advanced_tscv_report.md`
- `reports/black_advanced_tscv_report.md`
- `reports/model_selection_report.md`
- `reports/selected_model_tuning_report.md`
- `reports/final_model_serialization_report.md`
- `reports/results_comparison.csv`
- `reports/results_comparison_report.md`

### Documentação técnica

- `docs/decisoes_tecnicas.md`

> Arquivos `reports/*.json` e `models/*.joblib` são gerados durante a reprodução e ficam fora do versionamento (ver `.gitignore`).

## Como Interpretar Os Resultados

1. **Validade dos dados:** `reports/analytical_dataset_validation.json` confirma que o join e a persistência em Parquet fecharam sem perda inesperada; `reports/parquet_profiling_report.md` mostra nulos, duplicatas, cardinalidade e outliers; `reports/data_treatment_strategy.md` registra as regras de limpeza, imputação e capping.
2. **Estratégia temporal:** `reports/temporal_analysis.md` confirma a cobertura temporal, sazonalidade e gaps; `reports/tscv_strategy.md` documenta o desenho do `TimeSeriesSplit`.
3. **Seleção de modelo:** `reports/model_selection_report.md` consolida a comparação entre algoritmos; `reports/selected_model_tuning_report.md` mostra o ganho do tuning por categoria; `reports/results_comparison.csv` é o entregável principal de avaliação, com métricas por algoritmo, categoria e fold, além das médias finais.
4. **Deploy / uso final:** `reports/final_model_serialization_report.md` valida a serialização; os artefatos `models/*.joblib` podem ser carregados para inferência offline.

## Regras Críticas

- Não usar `price` da tabela `ride_v2.csv` como feature.
- Não usar campos de vazamento ou pós-evento como `RidePrice`, `Selected` e `RideReasonSelectedEstimativeID`.
- Todos os registros do mesmo `RideID` devem permanecer no mesmo fold.
- `Updated` não deve ser a âncora do treinamento.

## Limites Conhecidos

- O schema não traz timestamp por estimativa individual, então algumas features cruzadas usam `RideEstimativeID` como melhor proxy de ordem.
- O projeto ainda não inclui clima, trânsito, eventos locais ou sinais de oferta e demanda externa.
- Não existe API de inferência em produção nem monitoramento automatizado.

## Próximos Passos

1. Expor uma API de inferência.
2. Implementar monitoramento de drift e performance.
3. Automatizar retraining com janela temporal fixa.
4. Adicionar features externas como clima, eventos e trânsito.
5. Avaliar calibração, `log1p(target)` e interpretabilidade por SHAP.

## Documentação Complementar

- `data/README.md`
- `docs/decisoes_tecnicas.md`