# Decisoes Tecnicas, Limitacoes e Proximos Passos

## Visao Geral

Este documento resume as decisoes tecnicas consolidadas ao longo do projeto de previsao de preco por categoria. O foco foi manter um pipeline reprodutivel, temporalmente correto e com entregaveis comparaveis entre categorias.

## Decisoes Tecnicas Principais

### 1. Estrategia de dados e consolidacao

- A camada analitica foi persistida em Parquet particionado por `CategoryID` para leitura seletiva e melhor desempenho.
- A consolidacao final trabalha sobre a base `analytical_curated` e depois sobre as camadas `features_temporal` e `final_features`.
- O dataset final de modelagem foi separado por categoria: `UberX`, `Uber Comfort` e `Uber Black`.

### 2. Estrategia temporal e TSCV

- O timestamp de referencia para ordenacao e validacao temporal e `Create`.
- `Schedule` foi mantido como campo auxiliar para analise temporal, mas `Updated` nao foi usado como ancora do TSCV por apresentar cauda longa de atualizacao tardia.
- A estrategia final de validacao usa `TimeSeriesSplit` com:
  - `4` folds expansivos
  - `28` dias de validacao por fold
  - `7` dias de gap entre treino e validacao
  - holdout final com os ultimos `28` dias
- Regime modelado: `2021-11-01` em diante, para focar o periodo mais recente e com regime mais estavel.
- Todos os registros do mesmo `RideID` permanecem no mesmo fold para evitar leakage.

### 3. Tratamento de nulos, outliers e inconsistencias

- `FareID` recebeu sentinel categorico (`MISSING_FAREID`) e flag `FareIDWasImputed`.
- `WaitingTime` recebeu capping no percentil `99` e flag `WaitingTimeWasCapped`.
- `Price` recebeu capping superior por `CategoryID` no percentil `99.5`.
- Registros com `Price <= 0` foram removidos.
- Linhas com coordenadas invalidas ou ausentes foram removidas.
- Colunas com vazamento, baixa utilidade ou esparsidade extrema foram descartadas.

### 4. Features e engenharia

- Features temporais foram criadas a partir de `Create`, `Schedule` e `Updated`, mas apenas `Create` permaneceu como referencia temporal do treino.
- Features cruzadas de preco entre categorias foram criadas por `RideID`, sem duplicar registros.
- Agregacoes historicas por `UserID` foram calculadas com janela expansiva e `shift`, garantindo que somente informacao passada entrasse nas features.
- O conjunto final de features inclui variaveis temporais, espaciais, historicas, cruzadas e flags de qualidade.

### 5. Escolha de algoritmos

- Baselines: `LinearRegression`, `Ridge` e `Lasso`.
- Modelos avancados: `XGBoost`, `LightGBM` e `RandomForest`.
- O criterio de selecao final foi o menor `RMSE` medio no TSCV.
- Em todas as categorias, `LightGBM` foi o melhor modelo por `RMSE` medio.
- O tuning final foi feito com `Optuna` sobre o `LightGBM` selecionado.

### 6. Entregaveis finais

- `results_comparison.csv` consolidando metricas por algoritmo, categoria e fold.
- `best_model_by_category.csv` com a selecao final por categoria.
- Modelos finais serializados em `.joblib`:
  - `models/model_uberx.joblib`
  - `models/model_comfort.joblib`
  - `models/model_black.joblib`

## Limitacoes Conhecidas

- O schema nao traz timestamp por estimativa individual, entao as features cruzadas de preco usam `RideEstimativeID` como melhor proxy da ordem de disponibilidade.
- `Updated` nao e confiavel como eixo temporal principal porque pode ocorrer muito depois da corrida.
- `RidePrice`, `Selected` e `RideReasonSelectedEstimativeID` sao campos de vazamento ou pos-evento e ficam fora do treino.
- Algumas colunas foram descartadas por esparsidade extrema ou pouca variacao, mesmo sendo uteis para auditoria.
- O tuning mostrou ganhos desiguais entre metricas: em algumas categorias o `RMSE` melhorou enquanto `MAE` e `MAPE` tiveram ganho marginal ou leve regressao.
- O projeto ainda nao inclui variaveis externas como clima, eventos, feriados locais, transito ou disponibilidade de frota.
- Nao ha API de inferencia nem monitoramento automatizado em producao.

## Proximos Passos Sugeridos

1. Publicar uma API de inferencia para consumo dos modelos finais.
2. Implementar monitoramento de drift de dados e de performance.
3. Automatizar retraining com janela temporal fixa e gatilhos por degradacao.
4. Incluir features externas:
   - clima
   - eventos locais
   - feriados por municipio/estado
   - transito e mobilidade
   - sinais de oferta e demanda da frota
5. Explorar calibracao por categoria e transformacoes do target, como `log1p`, para reduzir impacto da cauda longa.
6. Testar validacao por periodo mais recente para confirmar estabilidade apos a quebra de regime observada em `nov/2021`.
7. Avaliar interpretabilidade com SHAP ou importancias por categoria para apoiar decisao de negocio.

## Referencias Internas

- Estrategia temporal: `reports/tscv_strategy.md`
- Tratamento de dados: `reports/data_treatment_strategy.md`
- Selecao de modelos: `reports/model_selection_report.md`
- Comparativo final: `reports/results_comparison_report.md`
- Serializacao final: `reports/final_model_serialization_report.md`

## Conclusao

O projeto fechou com um pipeline temporalmente consistente, features sem leakage conhecido no fluxo principal e um modelo final por categoria pronto para inferencia offline ou evolucao para producao.
