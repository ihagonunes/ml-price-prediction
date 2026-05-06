# Validacao das Features Temporais

- Gerado em: `2026-05-05T21:02:20.107315`
- Fonte: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\analytical_curated`
- Saida com features: `C:\Users\rodrigo.neiland\OneDrive - ESPM\Documentos\3sem\Martech\ml-price-prediction\data\features_temporal`
- Linhas processadas: `1999782`
- Linhas gravadas: `1999782`

## Features Criadas

- Para cada timestamp (`Schedule`, `Create`, `Updated`): `Hour`, `DayOfWeek`, `DayOfWeekName`, `Month`, `Quarter`, `IsHolidayBR`, `DayPeriod`.
- `DayPeriod` usa a regra de negocio: `madrugada` (00-05), `manha` (06-11), `tarde` (12-17) e `noite` (18-23).

## Decisao DS

- `CreateHour` e `CreateDayPeriod` sao prioritarias para transporte por capturarem picos de demanda e janelas de surge pricing.
- `CreateDayOfWeek` ajuda a separar dias uteis, sexta-feira e fim de semana, que costumam ter comportamento operacional diferente.
- `CreateMonth`, `CreateQuarter` e `CreateIsHolidayBR` capturam sazonalidade de calendario e mudancas de demanda em feriados nacionais.
- `Schedule*` foi mantida para cenarios de reserva/planejamento, embora no dataset atual ela seja muito proxima de `Create`.
- `Updated*` foi gerada por completude analitica, mas deve ficar fora do baseline de precificacao por risco de ser posterior ao momento de inferencia.

## Validacao DS

| feature         | value     |   row_count |   row_pct |
|:----------------|:----------|------------:|----------:|
| CreateDayPeriod | tarde     |      790723 |   39.5405 |
| CreateDayPeriod | manha     |      608982 |   30.4524 |
| CreateDayPeriod | noite     |      556807 |   27.8434 |
| CreateDayPeriod | madrugada |       43270 |    2.1637 |

| feature             | value   |   row_count |   row_pct |
|:--------------------|:--------|------------:|----------:|
| CreateDayOfWeekName | quinta  |      382791 |   19.1416 |
| CreateDayOfWeekName | quarta  |      377129 |   18.8585 |
| CreateDayOfWeekName | sexta   |      366710 |   18.3375 |
| CreateDayOfWeekName | terca   |      330588 |   16.5312 |
| CreateDayOfWeekName | segunda |      326118 |   16.3077 |
| CreateDayOfWeekName | sabado  |      197945 |    9.8983 |
| CreateDayOfWeekName | domingo |       18501 |    0.9252 |

- Horas de maior concentracao em `CreateHour`: 9h (11.51%), 10h (9.31%), 18h (8.79%), 19h (8.24%), 17h (7.27%).
- Domingo representa apenas `0.9252%` das linhas, coerente com a baixa atividade observada na EDA temporal.
- Feriados nacionais brasileiros representam `0.3035%` das linhas; a feature e rara, mas relevante para capturar choques pontuais de demanda.

## Distribuicao por Categoria-Alvo

|   CategoryID | CategoryName   | feature             | value     |   row_count |   row_pct |
|-------------:|:---------------|:--------------------|:----------|------------:|----------:|
|            2 | UberX          | CreateDayOfWeekName | quinta    |      133661 |   18.8243 |
|            2 | UberX          | CreateDayOfWeekName | quarta    |      132439 |   18.6522 |
|            2 | UberX          | CreateDayOfWeekName | sexta     |      131481 |   18.5173 |
|            2 | UberX          | CreateDayOfWeekName | segunda   |      117397 |   16.5337 |
|            2 | UberX          | CreateDayOfWeekName | terca     |      112998 |   15.9142 |
|            2 | UberX          | CreateDayOfWeekName | sabado    |       77176 |   10.8692 |
|            2 | UberX          | CreateDayOfWeekName | domingo   |        4894 |    0.6893 |
|            2 | UberX          | CreateDayPeriod     | tarde     |      281179 |   39.6001 |
|            2 | UberX          | CreateDayPeriod     | manha     |      232851 |   32.7938 |
|            2 | UberX          | CreateDayPeriod     | noite     |      184797 |   26.0261 |
|            2 | UberX          | CreateDayPeriod     | madrugada |       11219 |    1.58   |
|            2 | UberX          | CreateIsHolidayBR   | False     |      708332 |   99.7586 |
|            2 | UberX          | CreateIsHolidayBR   | True      |        1714 |    0.2414 |
|            4 | Uber Black     | CreateDayOfWeekName | quinta    |       24274 |   19.4222 |
|            4 | Uber Black     | CreateDayOfWeekName | quarta    |       23493 |   18.7973 |
|            4 | Uber Black     | CreateDayOfWeekName | sexta     |       22859 |   18.29   |
|            4 | Uber Black     | CreateDayOfWeekName | terca     |       20970 |   16.7786 |
|            4 | Uber Black     | CreateDayOfWeekName | segunda   |       20087 |   16.072  |
|            4 | Uber Black     | CreateDayOfWeekName | sabado    |       11913 |    9.5318 |
|            4 | Uber Black     | CreateDayOfWeekName | domingo   |        1385 |    1.1082 |
|            4 | Uber Black     | CreateDayPeriod     | tarde     |       50271 |   40.2229 |
|            4 | Uber Black     | CreateDayPeriod     | manha     |       37060 |   29.6525 |
|            4 | Uber Black     | CreateDayPeriod     | noite     |       34661 |   27.733  |
|            4 | Uber Black     | CreateDayPeriod     | madrugada |        2989 |    2.3916 |
|            4 | Uber Black     | CreateIsHolidayBR   | False     |      124519 |   99.6303 |
|            4 | Uber Black     | CreateIsHolidayBR   | True      |         462 |    0.3697 |
|            9 | Uber Comfort   | CreateDayOfWeekName | quinta    |       53185 |   19.4004 |
|            9 | Uber Comfort   | CreateDayOfWeekName | quarta    |       51986 |   18.963  |
|            9 | Uber Comfort   | CreateDayOfWeekName | sexta     |       50416 |   18.3903 |
|            9 | Uber Comfort   | CreateDayOfWeekName | terca     |       45410 |   16.5643 |
|            9 | Uber Comfort   | CreateDayOfWeekName | segunda   |       44177 |   16.1145 |
|            9 | Uber Comfort   | CreateDayOfWeekName | sabado    |       26431 |    9.6413 |
|            9 | Uber Comfort   | CreateDayOfWeekName | domingo   |        2539 |    0.9262 |
|            9 | Uber Comfort   | CreateDayPeriod     | tarde     |      107200 |   39.1035 |
|            9 | Uber Comfort   | CreateDayPeriod     | manha     |       85822 |   31.3054 |
|            9 | Uber Comfort   | CreateDayPeriod     | noite     |       75132 |   27.406  |
|            9 | Uber Comfort   | CreateDayPeriod     | madrugada |        5990 |    2.185  |
|            9 | Uber Comfort   | CreateIsHolidayBR   | False     |      273385 |   99.7231 |
|            9 | Uber Comfort   | CreateIsHolidayBR   | True      |         759 |    0.2769 |

## Conclusao

- As features temporais foram geradas de forma vetorizada e sem loops por linha.
- As distribuicoes ficaram coerentes com a EDA temporal anterior: concentracao em dias uteis e relevancia clara de granularidades intradia e semanais.