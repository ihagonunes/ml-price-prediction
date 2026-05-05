# Profiling do Dataset Analitico

- Gerado em: `2026-05-04T17:21:33.542635`
- Escopo: profiling reconstruido a partir do Parquet analitico, sem releitura dos CSVs.

## Resumo Executivo

### rideestimative

- Linhas logicas: `2000000`
- Colunas: `9`
- Nota de escopo: Tabela fato reconstruida diretamente do Parquet analitico, sem necessidade de deduplicacao adicional.
- Nulos relevantes: RideReasonSelectedEstimativeID (88.3%); FareID (40.91%)
- Duplicatas suspeitas: `PK duplicada=0` | `PK conflitante=0`
- Outliers relevantes: Price (6.31%); WaitingTime (4.73%)

### ride

- Linhas logicas: `239270`
- Colunas: `15`
- Nota de escopo: Tabela reconstruida a partir do Parquet analitico. Contem apenas RideIDs com pelo menos uma estimativa associada.
- Nulos relevantes: Car (99.68%); RideDriverLocationID (99.68%); RideProviderID (99.52%); ProviderID (99.44%); CategoryID (99.44%)
- Duplicatas suspeitas: `PK duplicada=0` | `PK conflitante=0`
- Outliers relevantes: price (5.75%)

### product

- Linhas logicas: `34`
- Colunas: `4`
- Nota de escopo: Tabela reconstruida a partir do Parquet analitico. Contem apenas ProductIDs efetivamente usados no dataset consolidado.
- Nulos relevantes: Nenhum achado relevante.
- Duplicatas suspeitas: `PK duplicada=0` | `PK conflitante=0`
- Outliers relevantes: Nenhum achado relevante.

## Interpretacao DS

### rideestimative

- Campos com nulos relevantes que pedem regra de imputacao ou analise de missingness: RideReasonSelectedEstimativeID, FareID.
- Nao ha duplicatas suspeitas na tabela logica reconstruida.
- Outliers iniciais detectados via IQR: Price (6.31% acima/abaixo de [-32.175, 94.985]); WaitingTime (4.73% acima/abaixo de [-0.5, 11.5]).

### ride

- Campos com nulos extremos e forte candidato a descarte ou uso muito restrito: Car, RideDriverLocationID, RideProviderID, ProviderID, CategoryID.
- Nao ha duplicatas suspeitas na tabela logica reconstruida.
- Outliers iniciais detectados via IQR: price (5.75% acima/abaixo de [-17.275, 60.765]).
- Colunas de alta cardinalidade devem ser tratadas como identificadores/temporais ou passar por encoding especifico: UserID, Schedule, Create, Updated.
- Colunas constantes e candidatas a remocao por nao agregarem variancia: ScheduledRide.
- Os campos Schedule, Create e Updated estao tipados como string no Parquet e devem ser convertidos para datetime antes do feature engineering.

### product

- Nao foram encontrados campos com nulos acima do limiar de alerta.
- Nao ha duplicatas suspeitas na tabela logica reconstruida.
- Nao houve sinal forte de outliers nas variaveis numericas elegiveis.
- Colunas de alta cardinalidade devem ser tratadas como identificadores/temporais ou passar por encoding especifico: Description.

## Detalhamento

### rideestimative

- Reconstrucao: {'rows_scanned_from_parquet': 2000000, 'distinct_rows_before_pk_dedup': 2000000, 'logical_rows_after_pk_dedup': 2000000, 'exact_duplicate_rows_in_parquet_view': 0, 'conflicting_primary_keys': 0}
- Tipos: {'RideEstimativeID': 'int64[pyarrow]', 'RideID': 'int64[pyarrow]', 'ProductID': 'string[pyarrow]', 'WaitingTime': 'int64[pyarrow]', 'Price': 'double[pyarrow]', 'FareID': 'string[pyarrow]', 'Selected': 'int64[pyarrow]', 'RideReasonSelectedEstimativeID': 'int64[pyarrow]', 'Fee': 'int64[pyarrow]'}
- Campos com nulos >= 30.0%: [{'column': 'RideReasonSelectedEstimativeID', 'null_pct': 88.3, 'null_count': 1765979}, {'column': 'FareID', 'null_pct': 40.91, 'null_count': 818225}]
- Cardinalidade categorica: {'ProductID': {'unique_count': 34, 'unique_ratio': 0.0, 'top_values': {'UberX': 235734, 'pop99': 234073, 'regular-taxi': 215867, 'turbo-taxi': 199214, 'Comfort': 192982, 'Flash': 130530, 'Black': 123677, 'poupa99': 115676, 'Uber Promo': 104139, 'Flash Moto': 95879}}, 'FareID': {'unique_count': 485, 'unique_ratio': 0.0004, 'top_values': {'84b66fec-3a48-4e62-b403-2fb9a420900a': 69840, 'c8af9fdf-1d48-4ecb-9eb0-ebc25cb281e8': 63049, '738ca8b3-affe-42d9-ba75-b2c5bba3b938': 62074, '7a400fa1-e332-4ccc-b081-fa0615ca8c8a': 60375, 'c0a6bfa8-a9fd-48a8-9183-7c0dedbf58a2': 59531, '6f0d38da-0d37-45db-a74b-1061c828a440': 42340, 'c7a42859-5a45-404f-85c9-c5aa1be11d15': 36641, 'd5ef01d9-7d54-413e-b265-425948d06e92': 24900, '85ad7c6e-1d74-4a8f-8a82-a6653d9eeb66': 24019, '04a804b5-4536-4951-906d-563e2d6ea249': 23914}}, 'Selected': {'unique_count': 2, 'unique_ratio': 0.0, 'top_values': {'0': 1765979, '1': 234021}}, 'RideReasonSelectedEstimativeID': {'unique_count': 2, 'unique_ratio': 0.0, 'top_values': {'1': 151321, '4': 82700}}, 'Fee': {'unique_count': 4, 'unique_ratio': 0.0, 'top_values': {'0': 1996489, '-20': 2620, '2': 633, '5': 258}}}
- Campos de alta cardinalidade: nenhum
- Colunas constantes: nenhuma
- Outliers (IQR): [{'column': 'Price', 'outlier_pct': 6.31, 'outlier_count': 126167}, {'column': 'WaitingTime', 'outlier_pct': 4.73, 'outlier_count': 94642}]

### ride

- Reconstrucao: {'rows_scanned_from_parquet': 2000000, 'distinct_rows_before_pk_dedup': 239270, 'logical_rows_after_pk_dedup': 239270, 'exact_duplicate_rows_in_parquet_view': 1760730, 'conflicting_primary_keys': 0}
- Tipos: {'RideID': 'int64[pyarrow]', 'UserID': 'string[pyarrow]', 'Schedule': 'string[pyarrow]', 'Create': 'string[pyarrow]', 'RideStatusID': 'int64[pyarrow]', 'CompanyID': 'int64[pyarrow]', 'ProviderID': 'int64[pyarrow]', 'RideProviderID': 'int64[pyarrow]', 'price': 'double[pyarrow]', 'Updated': 'string[pyarrow]', 'CategoryID': 'int64[pyarrow]', 'TotalUsers': 'int64[pyarrow]', 'Car': 'string[pyarrow]', 'RideDriverLocationID': 'int64[pyarrow]', 'ScheduledRide': 'int64[pyarrow]'}
- Campos com nulos >= 30.0%: [{'column': 'Car', 'null_pct': 99.68, 'null_count': 238497}, {'column': 'RideDriverLocationID', 'null_pct': 99.68, 'null_count': 238512}, {'column': 'RideProviderID', 'null_pct': 99.52, 'null_count': 238125}, {'column': 'ProviderID', 'null_pct': 99.44, 'null_count': 237928}, {'column': 'CategoryID', 'null_pct': 99.44, 'null_count': 237928}]
- Cardinalidade categorica: {'RideStatusID': {'unique_count': 3, 'unique_ratio': 0.0, 'top_values': {'2': 225122, '3': 13801, '9': 347}}, 'CompanyID': {'unique_count': 46, 'unique_ratio': 0.0002, 'top_values': {'40': 144453, '41': 26912, '52': 9987, '54': 9563, '69': 7904, '56': 6841, '64': 6026, '42': 5955, '75': 5769, '70': 2385}}, 'ProviderID': {'unique_count': 3, 'unique_ratio': 0.0022, 'top_values': {'3': 727, '5': 585, '7': 30}}, 'CategoryID': {'unique_count': 3, 'unique_ratio': 0.0022, 'top_values': {'2': 1251, '5': 90, '6': 1}}, 'TotalUsers': {'unique_count': 5, 'unique_ratio': 0.0, 'top_values': {'1': 236194, '2': 2267, '3': 614, '4': 194, '0': 1}}, 'Car': {'unique_count': 101, 'unique_ratio': 0.1307, 'top_values': {'GM - Chevrolet Onix': 42, 'GM - Chevrolet Prisma': 36, 'Renault Sandero': 30, 'VW Voyage': 28, 'LOGAN': 28, 'Renault Logan': 24, 'Fiat Argo': 24, 'ETIOS': 24, 'Nissan Versa': 23, 'SPIN': 23}}, 'ScheduledRide': {'unique_count': 1, 'unique_ratio': 0.0, 'top_values': {'0': 239270}}}
- Campos de alta cardinalidade: {'UserID': {'unique_count': 3699, 'unique_ratio': 0.0155}, 'Schedule': {'unique_count': 239269, 'unique_ratio': 1.0}, 'Create': {'unique_count': 239252, 'unique_ratio': 0.9999}, 'Updated': {'unique_count': 239262, 'unique_ratio': 1.0}}
- Colunas constantes: ['ScheduledRide']
- Outliers (IQR): [{'column': 'price', 'outlier_pct': 5.75, 'outlier_count': 13763}]

### product

- Reconstrucao: {'rows_scanned_from_parquet': 2000000, 'distinct_rows_before_pk_dedup': 34, 'logical_rows_after_pk_dedup': 34, 'exact_duplicate_rows_in_parquet_view': 1999966, 'conflicting_primary_keys': 0}
- Tipos: {'ProductID': 'string[pyarrow]', 'ProviderID': 'int64[pyarrow]', 'CategoryID': 'int64[pyarrow]', 'Description': 'string[pyarrow]'}
- Campos com nulos >= 30.0%: nenhum
- Cardinalidade categorica: {'ProviderID': {'unique_count': 4, 'unique_ratio': 0.1176, 'top_values': {'2': 20, '3': 9, '5': 4, '7': 1}}, 'CategoryID': {'unique_count': 8, 'unique_ratio': 0.2353, 'top_values': {'1': 10, '2': 7, '10': 6, '5': 4, '9': 3, '4': 2, '6': 1, '8': 1}}}
- Campos de alta cardinalidade: {'Description': {'unique_count': 33, 'unique_ratio': 0.9706}}
- Colunas constantes: nenhuma
- Outliers (IQR): nenhum
