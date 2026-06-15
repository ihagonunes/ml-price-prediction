# Modelo Preditivo de Precos por Categoria de Servico

**Area:** Martech / Analytics
**Time:** Data Science & Engenharia de Dados
**Status:** Concluido

## Objetivo

Prever o `Price` estimado das corridas em tres categorias independentes:

- `UberX`
- `Uber Comfort`
- `Uber Black`

O projeto usa as tabelas `ride_v2.csv`, `rideestimative_v3.csv`, `rideaddress_v1.csv` e `product.csv` como fonte, com pipeline temporalmente ordenado, engenharia de features e validacao por `TimeSeriesSplit`.

O campo alvo e `Price` da tabela `rideestimative_v3.csv`. O campo `price` da tabela `ride_v2.csv` e preco real e nao pode ser usado como feature.

## Estrutura do Repositorio

```text
ml-price-prediction/
  data/
    README.md
    ride_v2.csv
    rideestimative_v3.csv
    rideaddress_v1.csv
    product.csv
  docs/
    decisoes_tecnicas.md
  notebooks/
    eda_report_notes.ipynb
  src/
    ingestion.py
    profiling.py
    target_analysis.py
    correlation_analysis.py
    temporal_analysis.py
    data_treatment.py
    eda_report.py
    features.py
    export_final_features.py
    train.py
    train_advanced_uberx.py
    train_advanced_comfort.py
    train_advanced_black.py
    compare_model_results.py
    tune_selected_models.py
    serialize_final_models.py
    generate_results_comparison.py
    validate_notebooks.py
    model_artifact.py
  models/
  reports/
  requirements.txt
  README.md
```

## Como Obter os Dados

Os dados nao sao versionados neste repositorio.

1. Extraia o pacote original `.rar` do projeto.
2. Copie os arquivos para `data/` com os nomes esperados pelo pipeline:
   - `ride_v2.csv`
   - `rideestimative_v3.csv`
   - `rideaddress_v1.csv`
   - `product.csv`
3. Se voce tiver os arquivos com nomes legados (`ride.csv`, `rideestimative.csv`), renomeie-os ou ajuste as constantes em `src/ingestion.py`.

O arquivo `rideestimative_v3.csv` e grande e nao deve ser aberto no Excel. A leitura do pipeline usa `chunksize` para evitar estouro de memoria.

Veja tambem `data/README.md` para detalhes do layout esperado.

## Configuracao do Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

O projeto foi validado com Python 3.14 e as dependencias listadas em `requirements.txt`.

## Reproducao Completa Do Zero

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

Essa ordem produz toda a cadeia de artefatos: dados curados, features finais, modelos baseline, modelos avancados, tuning, serializacao final e consolidacao comparativa.

## Notebook De Apoio

Hoje existe apenas um notebook no repositorio:

- `notebooks/eda_report_notes.ipynb`

Ele e um notebook de anotacoes DS ligado ao `eda_report.html`. Para manter a reproducao limpa, use `Restart Kernel & Run All` ao reexecuta-lo, mesmo que ele seja atualmente apenas informativo.

## Onde Encontrar Os Artefatos

### Dados intermedios

- `data/analytical/` - dataset analitico consolidado em Parquet
- `data/analytical_curated/` - camada curada com tratamento de nulos, outliers e inconsistencias
- `data/features_temporal/` - features temporais e historicas por categoria
- `data/final_features/` - datasets finais por categoria prontos para treino

### Modelos

- `models/model_uberx.joblib`
- `models/model_comfort.joblib`
- `models/model_black.joblib`

### Principais relatorios

- `reports/analytical_dataset_validation.json`
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

### Documentacao tecnica

- `docs/decisoes_tecnicas.md`

## Como Interpretar Os Resultados

### 1. Validade dos dados

- `reports/analytical_dataset_validation.json` confirma que o join e a persistencia em Parquet fecharam sem perda inesperada.
- `reports/parquet_profiling_report.md` mostra nulos, duplicatas, cardinalidade e outliers.
- `reports/data_treatment_strategy.md` registra as regras de limpeza, imputacao e capping.

### 2. Estrategia temporal

- `reports/temporal_analysis.md` confirma a cobertura temporal, a sazonalidade e os gaps.
- `reports/tscv_strategy.md` documenta o desenho final do `TimeSeriesSplit`.
- `Create` e a ancora temporal do treinamento.

### 3. Selecao de modelo

- `reports/model_selection_report.md` consolida a comparacao entre algoritmos.
- `reports/selected_model_tuning_report.md` mostra o ganho do tuning por categoria.
- `reports/results_comparison.csv` e o entregavel principal de avaliacao, com metricas por algoritmo, categoria e fold, alem das medias finais.

O melhor modelo final por `RMSE` medio no TSCV foi `LightGBM_Tuned` nas tres categorias.

### 4. Deploy / uso final

- `reports/final_model_serialization_report.md` valida a serializacao dos modelos finais.
- Os artefatos `models/*.joblib` podem ser carregados para inferencia offline.

## Regras Criticas

- Nao usar `price` da tabela `ride_v2.csv` como feature.
- Nao usar campos de vazamento ou pos-evento como `RidePrice`, `Selected` e `RideReasonSelectedEstimativeID`.
- Todos os registros do mesmo `RideID` devem permanecer no mesmo fold.
- `Updated` nao deve ser a ancora do TSCV.

## Limites Conhecidos

- O schema nao traz timestamp por estimativa individual, entao algumas features cruzadas usam `RideEstimativeID` como melhor proxy de ordem.
- O projeto ainda nao inclui clima, transito, eventos locais ou sinais de oferta e demanda externa.
- Nao existe API de inferencia em producao nem monitoramento automatizado.

## Proximos Passos

1. Expor uma API de inferencia.
2. Implementar monitoramento de drift e performance.
3. Automatizar retraining com janela temporal fixa.
4. Adicionar features externas como clima, eventos e transito.
5. Avaliar calibracao, `log1p(target)` e interpretabilidade por SHAP.

## Documentacao Complementar

- `data/README.md`
- `docs/decisoes_tecnicas.md`
