from __future__ import annotations

import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from utils import cast_object_columns_to_string, parse_datetime_columns, prepare_output_dir


# ============================================================
# CONFIGURACAO
# ============================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
ANALYTICAL_DIR = DATA_DIR / "analytical"
SOURCE_DIR = DATA_DIR / "analytical_curated"
OUTPUT_DIR = DATA_DIR / "features_temporal"

DISTRIBUTION_FILE = REPORTS_DIR / "temporal_feature_distribution.csv"
CATEGORY_DISTRIBUTION_FILE = REPORTS_DIR / "temporal_feature_target_distribution.csv"
METRICS_FILE = REPORTS_DIR / "temporal_feature_metrics.csv"
REPORT_FILE = REPORTS_DIR / "temporal_feature_report.md"
CROSS_COVERAGE_FILE = REPORTS_DIR / "cross_price_feature_coverage.csv"
CROSS_REPORT_FILE = REPORTS_DIR / "cross_price_feature_report.md"
USER_HISTORY_COVERAGE_FILE = REPORTS_DIR / "user_history_feature_coverage.csv"
USER_HISTORY_REPORT_FILE = REPORTS_DIR / "user_history_feature_report.md"

BATCH_SIZE = 100_000
TIMESTAMP_COLUMNS = ["Schedule", "Create", "Updated"]
TARGET_CATEGORIES = [2, 9, 4]
TARGET_CATEGORY_NAMES = {
    2: "UberX",
    9: "Uber Comfort",
    4: "Uber Black",
}
CROSS_PRICE_PRODUCT_MAP = {
    "UberX": "Price_UberX",
    "Comfort": "Price_Comfort",
    "Black": "Price_Black",
}
CROSS_PRICE_COLUMNS = list(CROSS_PRICE_PRODUCT_MAP.values())
USER_HISTORY_FEATURES = [
    "UserPriorRideCount",
    "UserPriorPaidPriceMean",
    "UserPriorCategoryRideCount",
    "UserPriorCategoryPriceMean",
]
DAY_OF_WEEK_NAMES = {
    0: "segunda",
    1: "terca",
    2: "quarta",
    3: "quinta",
    4: "sexta",
    5: "sabado",
    6: "domingo",
}
DAY_PERIOD_ORDER = ["madrugada", "manha", "tarde", "noite"]
WEEKDAY_ORDER = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
VALIDATION_FEATURES = [
    "Hour",
    "DayOfWeekName",
    "Month",
    "Quarter",
    "IsHolidayBR",
    "DayPeriod",
]
CREATE_CATEGORY_FEATURES = [
    "CreateHour",
    "CreateDayOfWeekName",
    "CreateIsHolidayBR",
    "CreateDayPeriod",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def build_brazil_holiday_index() -> pd.DatetimeIndex:
    holiday_dates = holidays.country_holidays(
        "BR",
        years=range(2020, 2031),
    )
    return pd.DatetimeIndex(sorted(holiday_dates.keys())).normalize()


def build_day_period(hours: pd.Series) -> pd.Series:
    labels = np.select(
        [
            hours.between(0, 5),
            hours.between(6, 11),
            hours.between(12, 17),
            hours.between(18, 23),
        ],
        DAY_PERIOD_ORDER,
        default="desconhecido",
    )
    return pd.Series(labels, index=hours.index, dtype="string")


def add_temporal_features(
    df: pd.DataFrame,
    holiday_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    enriched = parse_datetime_columns(df, TIMESTAMP_COLUMNS)

    for column in TIMESTAMP_COLUMNS:
        timestamp_series = enriched[column]
        if timestamp_series.isna().any():
            raise ValueError(f"Existem timestamps nulos em {column}.")

        prefix = column
        enriched[f"{prefix}Hour"] = timestamp_series.dt.hour.astype("uint8")
        enriched[f"{prefix}DayOfWeek"] = timestamp_series.dt.dayofweek.astype("uint8")
        enriched[f"{prefix}DayOfWeekName"] = (
            enriched[f"{prefix}DayOfWeek"]
            .map(DAY_OF_WEEK_NAMES)
            .astype("string")
        )
        enriched[f"{prefix}Month"] = timestamp_series.dt.month.astype("uint8")
        enriched[f"{prefix}Quarter"] = timestamp_series.dt.quarter.astype("uint8")
        enriched[f"{prefix}IsHolidayBR"] = (
            timestamp_series.dt.normalize().isin(holiday_index).astype("bool")
        )
        enriched[f"{prefix}DayPeriod"] = build_day_period(
            enriched[f"{prefix}Hour"].astype("int16")
        )

    return cast_object_columns_to_string(enriched)


def build_cross_price_lookup(
    dataset: ds.Dataset,
) -> tuple[pd.DataFrame, dict]:
    lookup_df = dataset.to_table(
        columns=["RideID", "RideEstimativeID", "ProductID", "Price"]
    ).to_pandas()
    lookup_df["ProductID"] = lookup_df["ProductID"].astype("string")
    lookup_df = lookup_df.loc[
        lookup_df["ProductID"].isin(CROSS_PRICE_PRODUCT_MAP)
    ].copy()
    lookup_df = lookup_df.sort_values(
        by=["RideID", "ProductID", "RideEstimativeID"],
        kind="stable",
    )

    duplicate_mask = lookup_df.duplicated(["RideID", "ProductID"], keep="first")
    duplicate_rows = lookup_df.loc[duplicate_mask].copy()
    canonical_lookup_df = lookup_df.loc[~duplicate_mask].copy()

    pivot_df = canonical_lookup_df.pivot(
        index="RideID",
        columns="ProductID",
        values="Price",
    ).rename(columns=CROSS_PRICE_PRODUCT_MAP)
    pivot_df = pivot_df.reindex(columns=CROSS_PRICE_COLUMNS)
    pivot_df = pivot_df.reset_index()

    metrics = {
        "canonical_source_rows": int(len(lookup_df)),
        "canonical_unique_ride_ids": int(lookup_df["RideID"].nunique()),
        "canonical_duplicate_rows_collapsed": int(duplicate_mask.sum()),
        "rides_with_all_three_canonical_prices": int(
            pivot_df[CROSS_PRICE_COLUMNS].notna().all(axis=1).sum()
        ),
    }

    for product_name, feature_name in CROSS_PRICE_PRODUCT_MAP.items():
        product_rows = lookup_df.loc[lookup_df["ProductID"] == product_name]
        metrics[f"{product_name}_rows"] = int(len(product_rows))
        metrics[f"{product_name}_unique_rides"] = int(product_rows["RideID"].nunique())
        metrics[f"{product_name}_duplicates_collapsed"] = int(
            duplicate_rows["ProductID"].eq(product_name).sum()
        )
        metrics[f"{feature_name}_ride_coverage_pct"] = round(
            pivot_df[feature_name].notna().mean() * 100,
            4,
        )

    return pivot_df, metrics


def add_cross_price_features(
    df: pd.DataFrame,
    cross_price_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = df.merge(
        cross_price_lookup_df,
        on="RideID",
        how="left",
        validate="many_to_one",
    )
    return enriched


def build_user_ride_history_lookup(
    curated_dataset: ds.Dataset,
    analytical_dataset: ds.Dataset,
) -> tuple[pd.DataFrame, dict]:
    ride_base_df = curated_dataset.to_table(
        columns=["RideID", "UserID", "Create"]
    ).to_pandas()
    raw_ride_rows = len(ride_base_df)
    ride_base_df = (
        ride_base_df.sort_values(["RideID", "Create"], kind="stable")
        .drop_duplicates(["RideID"], keep="first")
        .reset_index(drop=True)
    )

    ride_price_df = analytical_dataset.to_table(
        columns=["RideID", "RidePrice"]
    ).to_pandas()
    ride_price_df = (
        ride_price_df.sort_values(["RideID"], kind="stable")
        .drop_duplicates(["RideID"], keep="first")
        .reset_index(drop=True)
    )

    ride_history_df = ride_base_df.merge(
        ride_price_df,
        on="RideID",
        how="left",
        validate="one_to_one",
    )
    ride_history_df["Create"] = pd.to_datetime(ride_history_df["Create"], errors="coerce")
    ride_history_df = ride_history_df.sort_values(
        ["UserID", "Create", "RideID"],
        kind="stable",
    ).reset_index(drop=True)

    ride_history_df["UserPriorRideCount"] = (
        ride_history_df.groupby("UserID", sort=False)
        .cumcount()
        .astype("int32")
    )

    paid_price = ride_history_df["RidePrice"].where(ride_history_df["RidePrice"] > 0)
    paid_value = paid_price.fillna(0.0)
    paid_flag = paid_price.notna().astype("int32")
    ride_history_df["_paid_value"] = paid_value
    ride_history_df["_paid_flag"] = paid_flag
    grouped_user = ride_history_df.groupby("UserID", sort=False)
    prior_paid_sum = grouped_user["_paid_value"].cumsum() - ride_history_df["_paid_value"]
    prior_paid_count = grouped_user["_paid_flag"].cumsum() - ride_history_df["_paid_flag"]
    ride_history_df["UserPriorPaidPriceMean"] = np.where(
        prior_paid_count > 0,
        prior_paid_sum / prior_paid_count,
        np.nan,
    ).astype("float32")
    ride_history_df = ride_history_df.drop(columns=["Create", "RidePrice", "_paid_value", "_paid_flag"])

    lookup_df = ride_history_df[
        ["RideID", "UserPriorRideCount", "UserPriorPaidPriceMean"]
    ].copy()
    metrics = {
        "user_history_ride_rows_scanned": int(raw_ride_rows),
        "user_history_unique_rides": int(len(lookup_df)),
        "user_history_unique_users": int(ride_base_df["UserID"].nunique()),
        "user_history_duplicate_ride_rows_collapsed": int(raw_ride_rows - len(lookup_df)),
        "user_history_rides_with_valid_paid_price": int((paid_price.notna()).sum()),
        "user_history_rows_with_prior_paid_mean": int(lookup_df["UserPriorPaidPriceMean"].notna().sum()),
        "user_history_cold_start_rides": int((lookup_df["UserPriorRideCount"] == 0).sum()),
        "user_history_prior_ride_count_p50": int(lookup_df["UserPriorRideCount"].quantile(0.50)),
        "user_history_prior_ride_count_p90": int(lookup_df["UserPriorRideCount"].quantile(0.90)),
        "user_history_prior_ride_count_p99": int(lookup_df["UserPriorRideCount"].quantile(0.99)),
    }
    return lookup_df, metrics


def build_user_category_history_lookup(
    curated_dataset: ds.Dataset,
) -> tuple[pd.DataFrame, dict]:
    category_base_df = curated_dataset.to_table(
        columns=["RideID", "RideEstimativeID", "UserID", "Create", "CategoryID", "Price"]
    ).to_pandas()
    raw_category_rows = len(category_base_df)
    category_base_df["Create"] = pd.to_datetime(category_base_df["Create"], errors="coerce")
    category_base_df = category_base_df.sort_values(
        ["RideID", "CategoryID", "RideEstimativeID"],
        kind="stable",
    )
    canonical_category_df = (
        category_base_df
        .drop_duplicates(["RideID", "CategoryID"], keep="first")
        .sort_values(
            ["UserID", "CategoryID", "Create", "RideID", "RideEstimativeID"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    category_group = canonical_category_df.groupby(["UserID", "CategoryID"], sort=False)
    canonical_category_df["UserPriorCategoryRideCount"] = (
        category_group.cumcount().astype("int32")
    )
    prior_category_sum = category_group["Price"].cumsum() - canonical_category_df["Price"]
    canonical_category_df["UserPriorCategoryPriceMean"] = np.where(
        canonical_category_df["UserPriorCategoryRideCount"] > 0,
        prior_category_sum / canonical_category_df["UserPriorCategoryRideCount"],
        np.nan,
    ).astype("float32")

    lookup_df = canonical_category_df[
        ["RideID", "CategoryID", "UserPriorCategoryRideCount", "UserPriorCategoryPriceMean"]
    ].copy()
    metrics = {
        "user_category_history_rows_scanned": int(raw_category_rows),
        "user_category_history_unique_ride_category_pairs": int(len(lookup_df)),
        "user_category_history_duplicate_rows_collapsed": int(raw_category_rows - len(lookup_df)),
        "user_category_history_rows_with_prior_category_mean": int(
            lookup_df["UserPriorCategoryPriceMean"].notna().sum()
        ),
        "user_category_history_cold_start_pairs": int(
            (lookup_df["UserPriorCategoryRideCount"] == 0).sum()
        ),
        "user_category_history_prior_count_p50": int(
            lookup_df["UserPriorCategoryRideCount"].quantile(0.50)
        ),
        "user_category_history_prior_count_p90": int(
            lookup_df["UserPriorCategoryRideCount"].quantile(0.90)
        ),
        "user_category_history_prior_count_p99": int(
            lookup_df["UserPriorCategoryRideCount"].quantile(0.99)
        ),
    }
    return lookup_df, metrics


def add_user_history_features(
    df: pd.DataFrame,
    user_ride_history_lookup_df: pd.DataFrame,
    user_category_history_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = df.merge(
        user_ride_history_lookup_df,
        on="RideID",
        how="left",
        validate="many_to_one",
    )
    enriched = enriched.merge(
        user_category_history_lookup_df,
        on=["RideID", "CategoryID"],
        how="left",
        validate="many_to_one",
    )
    if enriched["UserPriorRideCount"].isna().any():
        raise ValueError("Falha ao popular UserPriorRideCount para todas as linhas.")
    if enriched["UserPriorCategoryRideCount"].isna().any():
        raise ValueError("Falha ao popular UserPriorCategoryRideCount para todas as linhas.")

    enriched["UserPriorRideCount"] = enriched["UserPriorRideCount"].astype("int32")
    enriched["UserPriorCategoryRideCount"] = enriched["UserPriorCategoryRideCount"].astype("int32")
    enriched["UserPriorPaidPriceMean"] = enriched["UserPriorPaidPriceMean"].astype("float32")
    enriched["UserPriorCategoryPriceMean"] = enriched["UserPriorCategoryPriceMean"].astype("float32")
    return enriched


def update_overall_distribution(
    df: pd.DataFrame,
    counter: Counter,
) -> None:
    for column in TIMESTAMP_COLUMNS:
        for suffix in VALIDATION_FEATURES:
            feature_name = f"{column}{suffix}"
            counts = (
                df[feature_name]
                .astype("string")
                .value_counts(dropna=False)
                .to_dict()
            )
            for value, row_count in counts.items():
                counter[(feature_name, str(value))] += int(row_count)


def update_category_distribution(
    df: pd.DataFrame,
    counter: Counter,
) -> None:
    target_df = df.loc[df["CategoryID"].isin(TARGET_CATEGORIES)].copy()
    if target_df.empty:
        return

    for feature_name in CREATE_CATEGORY_FEATURES:
        grouped = (
            target_df.groupby(["CategoryID", feature_name], dropna=False)
            .size()
            .to_dict()
        )
        for (category_id, feature_value), row_count in grouped.items():
            counter[(int(category_id), feature_name, str(feature_value))] += int(row_count)


def build_overall_distribution_df(
    counter: Counter,
    total_rows: int,
) -> pd.DataFrame:
    rows = [
        {
            "feature": feature,
            "value": value,
            "row_count": row_count,
            "row_pct": round((row_count / total_rows) * 100, 4),
        }
        for (feature, value), row_count in sorted(counter.items())
    ]
    distribution_df = pd.DataFrame(rows)
    return distribution_df.sort_values(
        by=["feature", "row_count", "value"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_category_distribution_df(
    counter: Counter,
    category_totals: dict[int, int],
) -> pd.DataFrame:
    rows = []
    for (category_id, feature, value), row_count in sorted(counter.items()):
        category_total = category_totals[category_id]
        rows.append(
            {
                "CategoryID": category_id,
                "CategoryName": TARGET_CATEGORY_NAMES[category_id],
                "feature": feature,
                "value": value,
                "row_count": row_count,
                "row_pct": round((row_count / category_total) * 100, 4),
            }
        )

    distribution_df = pd.DataFrame(rows)
    return distribution_df.sort_values(
        by=["CategoryID", "feature", "row_count", "value"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_cross_feature_coverage_df(
    model_metrics: list[dict],
) -> pd.DataFrame:
    coverage_df = pd.DataFrame(model_metrics)
    return coverage_df.sort_values(
        by=["ModelProduct", "FeatureRole", "FeatureName"],
        kind="stable",
    ).reset_index(drop=True)


def build_user_history_coverage_df(
    model_metrics: list[dict],
) -> pd.DataFrame:
    coverage_df = pd.DataFrame(model_metrics)
    return coverage_df.sort_values(
        by=["CategoryID", "FeatureRole", "FeatureName"],
        kind="stable",
    ).reset_index(drop=True)


def write_validation_report(
    metrics: dict,
    overall_distribution_df: pd.DataFrame,
    category_distribution_df: pd.DataFrame,
    cross_coverage_df: pd.DataFrame,
    user_history_coverage_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overall_distribution_df.to_csv(DISTRIBUTION_FILE, index=False)
    category_distribution_df.to_csv(CATEGORY_DISTRIBUTION_FILE, index=False)
    cross_coverage_df.to_csv(CROSS_COVERAGE_FILE, index=False)
    user_history_coverage_df.to_csv(USER_HISTORY_COVERAGE_FILE, index=False)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(METRICS_FILE, index=False)

    create_features = category_distribution_df.loc[
        category_distribution_df["feature"].isin(
            ["CreateDayPeriod", "CreateDayOfWeekName", "CreateIsHolidayBR"]
        )
    ].copy()
    create_period_df = overall_distribution_df.loc[
        overall_distribution_df["feature"] == "CreateDayPeriod"
    ].copy()
    create_weekday_df = overall_distribution_df.loc[
        overall_distribution_df["feature"] == "CreateDayOfWeekName"
    ].copy()
    create_hour_df = overall_distribution_df.loc[
        overall_distribution_df["feature"] == "CreateHour"
    ].copy()
    create_holiday_df = overall_distribution_df.loc[
        overall_distribution_df["feature"] == "CreateIsHolidayBR"
    ].copy()

    top_hours = ", ".join(
        [
            f"{row.value}h ({row.row_pct:.2f}%)"
            for row in create_hour_df.head(5).itertuples(index=False)
        ]
    )
    sunday_pct = float(
        create_weekday_df.loc[
            create_weekday_df["value"] == "domingo",
            "row_pct",
        ].iloc[0]
    )
    holiday_pct = float(
        create_holiday_df.loc[
            create_holiday_df["value"] == "True",
            "row_pct",
        ].iloc[0]
    )

    report_lines = [
        "# Validacao das Features Temporais",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{SOURCE_DIR}`",
        f"- Saida com features: `{OUTPUT_DIR}`",
        f"- Linhas processadas: `{metrics['rows_processed']}`",
        f"- Linhas gravadas: `{metrics['rows_written']}`",
        "",
        "## Features Criadas",
        "",
        "- Para cada timestamp (`Schedule`, `Create`, `Updated`): `Hour`, `DayOfWeek`, `DayOfWeekName`, `Month`, `Quarter`, `IsHolidayBR`, `DayPeriod`.",
        "- `DayPeriod` usa a regra de negocio: `madrugada` (00-05), `manha` (06-11), `tarde` (12-17) e `noite` (18-23).",
        "",
        "## Decisao DS",
        "",
        "- `CreateHour` e `CreateDayPeriod` sao prioritarias para transporte por capturarem picos de demanda e janelas de surge pricing.",
        "- `CreateDayOfWeek` ajuda a separar dias uteis, sexta-feira e fim de semana, que costumam ter comportamento operacional diferente.",
        "- `CreateMonth`, `CreateQuarter` e `CreateIsHolidayBR` capturam sazonalidade de calendario e mudancas de demanda em feriados nacionais.",
        "- `Schedule*` foi mantida para cenarios de reserva/planejamento, embora no dataset atual ela seja muito proxima de `Create`.",
        "- `Updated*` foi gerada por completude analitica, mas deve ficar fora do baseline de precificacao por risco de ser posterior ao momento de inferencia.",
        "",
        "## Validacao DS",
        "",
        create_period_df.to_markdown(index=False),
        "",
        create_weekday_df.to_markdown(index=False),
        "",
        f"- Horas de maior concentracao em `CreateHour`: {top_hours}.",
        f"- Domingo representa apenas `{sunday_pct:.4f}%` das linhas, coerente com a baixa atividade observada na EDA temporal.",
        f"- Feriados nacionais brasileiros representam `{holiday_pct:.4f}%` das linhas; a feature e rara, mas relevante para capturar choques pontuais de demanda.",
        "",
        "## Distribuicao por Categoria-Alvo",
        "",
        create_features.to_markdown(index=False),
        "",
        "## Conclusao",
        "",
        "- As features temporais foram geradas de forma vetorizada e sem loops por linha.",
        "- As distribuicoes ficaram coerentes com a EDA temporal anterior: concentracao em dias uteis e relevancia clara de granularidades intradia e semanais.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    cross_report_lines = [
        "# Features Cruzadas de Preco entre Categorias",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{SOURCE_DIR}`",
        f"- Saida com features: `{OUTPUT_DIR}`",
        "",
        "## Regra de Pivot",
        "",
        "- O pivot usa apenas os produtos canonicos `UberX`, `Comfort` e `Black` para gerar `Price_UberX`, `Price_Comfort` e `Price_Black`.",
        "- A chave do lookup e `RideID`, com deduplicacao rara em `RideID + ProductID` pelo menor `RideEstimativeID`, para privilegiar a primeira estimativa disponivel e evitar usar refreshes posteriores.",
        "- O merge de volta na base principal e `many-to-one` por `RideID`, sem aumento do numero de linhas.",
        "",
        "## Regra DS de Uso sem Leakage",
        "",
        "- No modelo `UberX`, usar `Price_Comfort` e `Price_Black` como auxiliares e excluir `Price_UberX` da matriz de features.",
        "- No modelo `Comfort`, usar `Price_UberX` e `Price_Black` como auxiliares e excluir `Price_Comfort`.",
        "- No modelo `Black`, usar `Price_UberX` e `Price_Comfort` como auxiliares e excluir `Price_Black`.",
        "- Como o schema nao traz timestamp proprio por estimativa, `RideEstimativeID` foi usado como melhor proxy de ordem dentro da mesma corrida.",
        "",
        "## Cobertura das Features Cruzadas",
        "",
        cross_coverage_df.to_markdown(index=False),
        "",
        "## Integridade",
        "",
        f"- Duplicatas canonicas colapsadas no pivot: `{metrics['canonical_duplicate_rows_collapsed']}`.",
        f"- Corridas com as tres estimativas canonicas disponiveis: `{metrics['rides_with_all_three_canonical_prices']}`.",
        f"- Linhas antes/depois do join: `{metrics['rows_processed']}` / `{metrics['rows_written']}`.",
    ]
    CROSS_REPORT_FILE.write_text(
        "\n".join(cross_report_lines),
        encoding="utf-8",
    )

    user_history_report_lines = [
        "# Features Historicas por Usuario",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte principal: `{SOURCE_DIR}`",
        f"- Fonte de preco pago historico: `{ANALYTICAL_DIR}`",
        f"- Saida com features: `{OUTPUT_DIR}`",
        "",
        "## Features Criadas",
        "",
        "- `UserPriorRideCount`: quantidade de corridas anteriores do usuario no nivel de `RideID`.",
        "- `UserPriorPaidPriceMean`: media expansiva do `RidePrice` das corridas anteriores do usuario, excluindo o registro atual.",
        "- `UserPriorCategoryRideCount`: quantidade de ocorrencias anteriores do usuario na mesma `CategoryID`, calculada no nivel canonico de `RideID + CategoryID`.",
        "- `UserPriorCategoryPriceMean`: ticket medio historico do usuario para a mesma `CategoryID`, usando o `Price` canonico anterior e excluindo a corrida atual.",
        "",
        "## Regra Anti-Leakage",
        "",
        "- A ancora temporal e `Create`, que ja foi validada como o melhor indice temporal do projeto.",
        "- As features gerais do usuario sao calculadas no nivel de `RideID`, evitando que varias estimativas da mesma corrida pesem mais de uma vez.",
        "- As features por categoria usam uma versao canonica de `RideID + CategoryID`, deduplicada pelo menor `RideEstimativeID`, para evitar leakage entre ofertas repetidas da mesma corrida.",
        "- Em ambos os casos, a janela e expansiva e exclui o registro corrente via `cumcount` e `cumsum` com deslocamento logico.",
        "",
        "## Decisao DS",
        "",
        "- `UserPriorRideCount` captura maturidade e frequencia de uso do cliente.",
        "- `UserPriorPaidPriceMean` aproxima o ticket historico real do usuario e ajuda a separar perfis sensiveis a preco de perfis premium.",
        "- `UserPriorCategoryRideCount` e `UserPriorCategoryPriceMean` trazem preferencia e faixa de preco historica por categoria, sinal especialmente util para `UberX`, `Comfort` e `Black`.",
        "",
        "## Cobertura por Categoria-Alvo",
        "",
        user_history_coverage_df.to_markdown(index=False),
        "",
        "## Integridade",
        "",
        f"- Linhas antes/depois do pipeline: `{metrics['rows_processed']}` / `{metrics['rows_written']}`.",
        f"- Duplicatas colapsadas no historico por corrida: `{metrics['user_history_duplicate_ride_rows_collapsed']}`.",
        f"- Duplicatas colapsadas no historico por `RideID + CategoryID`: `{metrics['user_category_history_duplicate_rows_collapsed']}`.",
        f"- Cold start no nivel de corrida: `{metrics['user_history_cold_start_rides']}` corridas unicas.",
        f"- Cold start no nivel de categoria: `{metrics['user_category_history_cold_start_pairs']}` pares canonicos `RideID + CategoryID`.",
    ]
    USER_HISTORY_REPORT_FILE.write_text(
        "\n".join(user_history_report_lines),
        encoding="utf-8",
    )
    log.info("Relatorios salvos em %s", REPORTS_DIR)


# ============================================================
# PIPELINE
# ============================================================

def main() -> dict:
    log.info("=" * 60)
    log.info("Gerando camada de features")
    log.info("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset curado nao encontrado em {SOURCE_DIR}")
    if not ANALYTICAL_DIR.exists():
        raise FileNotFoundError(f"Dataset analitico nao encontrado em {ANALYTICAL_DIR}")

    prepare_output_dir(OUTPUT_DIR, DATA_DIR)

    dataset = ds.dataset(
        SOURCE_DIR,
        format="parquet",
        partitioning="hive",
    )
    analytical_dataset = ds.dataset(
        ANALYTICAL_DIR,
        format="parquet",
        partitioning="hive",
    )
    holiday_index = build_brazil_holiday_index()
    cross_price_lookup_df, cross_price_metrics = build_cross_price_lookup(dataset)
    user_ride_history_lookup_df, user_ride_history_metrics = build_user_ride_history_lookup(
        dataset,
        analytical_dataset,
    )
    user_category_history_lookup_df, user_category_history_metrics = (
        build_user_category_history_lookup(dataset)
    )

    metrics = {
        "rows_processed": 0,
        "rows_written": 0,
        "chunks_written": 0,
        "rows_on_disk": 0,
    }
    metrics.update(cross_price_metrics)
    metrics.update(user_ride_history_metrics)
    metrics.update(user_category_history_metrics)
    partition_row_counts: Counter = Counter()
    overall_counter: Counter = Counter()
    category_counter: Counter = Counter()
    target_category_totals = {category_id: 0 for category_id in TARGET_CATEGORIES}
    cross_model_stats = {
        product_name: {
            "row_count": 0,
            "both_cross_available": 0,
            **{feature_name: 0 for feature_name in CROSS_PRICE_COLUMNS},
        }
        for product_name in CROSS_PRICE_PRODUCT_MAP
    }
    user_history_model_stats = {
        category_id: {
            "row_count": 0,
            "prior_ride_rows": 0,
            "prior_paid_mean_rows": 0,
            "prior_category_rows": 0,
            "prior_category_mean_rows": 0,
        }
        for category_id in TARGET_CATEGORIES
    }
    user_history_overall_stats = {
        "prior_ride_rows": 0,
        "prior_paid_mean_rows": 0,
        "prior_category_rows": 0,
        "prior_category_mean_rows": 0,
    }
    partitioning = None

    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        batch_df = batch.to_pandas()
        metrics["rows_processed"] += len(batch_df)

        enriched_df = add_temporal_features(batch_df, holiday_index)
        enriched_df = add_cross_price_features(enriched_df, cross_price_lookup_df)
        enriched_df = add_user_history_features(
            enriched_df,
            user_ride_history_lookup_df,
            user_category_history_lookup_df,
        )
        update_overall_distribution(enriched_df, overall_counter)
        update_category_distribution(enriched_df, category_counter)

        user_history_overall_stats["prior_ride_rows"] += int(
            enriched_df["UserPriorRideCount"].gt(0).sum()
        )
        user_history_overall_stats["prior_paid_mean_rows"] += int(
            enriched_df["UserPriorPaidPriceMean"].notna().sum()
        )
        user_history_overall_stats["prior_category_rows"] += int(
            enriched_df["UserPriorCategoryRideCount"].gt(0).sum()
        )
        user_history_overall_stats["prior_category_mean_rows"] += int(
            enriched_df["UserPriorCategoryPriceMean"].notna().sum()
        )

        for category_id in TARGET_CATEGORIES:
            category_mask = enriched_df["CategoryID"].eq(category_id)
            category_row_count = int(category_mask.sum())
            target_category_totals[category_id] += category_row_count
            if not category_row_count:
                continue

            category_df = enriched_df.loc[category_mask]
            user_history_model_stats[category_id]["row_count"] += category_row_count
            user_history_model_stats[category_id]["prior_ride_rows"] += int(
                category_df["UserPriorRideCount"].gt(0).sum()
            )
            user_history_model_stats[category_id]["prior_paid_mean_rows"] += int(
                category_df["UserPriorPaidPriceMean"].notna().sum()
            )
            user_history_model_stats[category_id]["prior_category_rows"] += int(
                category_df["UserPriorCategoryRideCount"].gt(0).sum()
            )
            user_history_model_stats[category_id]["prior_category_mean_rows"] += int(
                category_df["UserPriorCategoryPriceMean"].notna().sum()
            )

        for product_name, own_feature in CROSS_PRICE_PRODUCT_MAP.items():
            model_df = enriched_df.loc[enriched_df["ProductID"] == product_name].copy()
            if model_df.empty:
                continue

            cross_model_stats[product_name]["row_count"] += int(len(model_df))
            other_feature_names = [
                feature_name
                for feature_name in CROSS_PRICE_COLUMNS
                if feature_name != own_feature
            ]
            for feature_name in CROSS_PRICE_COLUMNS:
                cross_model_stats[product_name][feature_name] += int(
                    model_df[feature_name].notna().sum()
                )
            cross_model_stats[product_name]["both_cross_available"] += int(
                model_df[other_feature_names].notna().all(axis=1).sum()
            )

        partition_counts = (
            enriched_df["CategoryID"]
            .astype("int64")
            .astype("string")
            .value_counts()
            .to_dict()
        )
        partition_row_counts.update(partition_counts)

        table = pa.Table.from_pandas(enriched_df, preserve_index=False)
        if partitioning is None:
            partitioning = ds.partitioning(
                pa.schema([("CategoryID", table.schema.field("CategoryID").type)]),
                flavor="hive",
            )

        metrics["chunks_written"] += 1
        ds.write_dataset(
            table,
            base_dir=OUTPUT_DIR,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"part-{metrics['chunks_written']:05d}-{{i}}.parquet",
        )
        metrics["rows_written"] += len(enriched_df)
        log.info(
            "Chunk com features #%s salvo | linhas=%s | acumulado=%s",
            metrics["chunks_written"],
            len(enriched_df),
            metrics["rows_written"],
        )

    featured_dataset = ds.dataset(
        OUTPUT_DIR,
        format="parquet",
        partitioning="hive",
    )
    metrics["rows_on_disk"] = int(featured_dataset.count_rows())
    if metrics["rows_on_disk"] != metrics["rows_written"]:
        raise ValueError(
            "Volume em disco diferente do processado. "
            f"Disco={metrics['rows_on_disk']} | Processado={metrics['rows_written']}"
        )

    metrics["partition_count"] = len(partition_row_counts)
    for category_id, row_count in sorted(partition_row_counts.items(), key=lambda item: int(item[0])):
        metrics[f"partition_{category_id}_rows"] = int(row_count)

    for metric_name, row_count in user_history_overall_stats.items():
        metrics[metric_name] = int(row_count)
        metrics[f"{metric_name}_pct"] = round(
            (row_count / metrics["rows_written"]) * 100,
            4,
        )

    cross_coverage_rows = []
    for product_name, own_feature in CROSS_PRICE_PRODUCT_MAP.items():
        row_count = cross_model_stats[product_name]["row_count"]
        for feature_name in CROSS_PRICE_COLUMNS:
            cross_coverage_rows.append(
                {
                    "ModelProduct": product_name,
                    "FeatureName": feature_name,
                    "FeatureRole": (
                        "target_equivalent"
                        if feature_name == own_feature
                        else "auxiliary_input"
                    ),
                    "available_rows": cross_model_stats[product_name][feature_name],
                    "available_pct": round(
                        (
                            cross_model_stats[product_name][feature_name]
                            / row_count
                            * 100
                        )
                        if row_count
                        else 0,
                        4,
                    ),
                }
            )
        cross_coverage_rows.append(
            {
                "ModelProduct": product_name,
                "FeatureName": "Both auxiliary prices",
                "FeatureRole": "auxiliary_pair",
                "available_rows": cross_model_stats[product_name]["both_cross_available"],
                "available_pct": round(
                    (
                        cross_model_stats[product_name]["both_cross_available"]
                        / row_count
                        * 100
                    )
                    if row_count
                    else 0,
                    4,
                ),
            }
        )

    user_history_coverage_rows = []
    for category_id in TARGET_CATEGORIES:
        row_count = user_history_model_stats[category_id]["row_count"]
        category_name = TARGET_CATEGORY_NAMES[category_id]
        user_history_coverage_rows.extend(
            [
                {
                    "CategoryID": category_id,
                    "CategoryName": category_name,
                    "FeatureName": "UserPriorRideCount",
                    "FeatureRole": "history_count",
                    "available_rows": user_history_model_stats[category_id]["prior_ride_rows"],
                    "available_pct": round(
                        (
                            user_history_model_stats[category_id]["prior_ride_rows"]
                            / row_count
                            * 100
                        )
                        if row_count
                        else 0,
                        4,
                    ),
                },
                {
                    "CategoryID": category_id,
                    "CategoryName": category_name,
                    "FeatureName": "UserPriorPaidPriceMean",
                    "FeatureRole": "history_value",
                    "available_rows": user_history_model_stats[category_id]["prior_paid_mean_rows"],
                    "available_pct": round(
                        (
                            user_history_model_stats[category_id]["prior_paid_mean_rows"]
                            / row_count
                            * 100
                        )
                        if row_count
                        else 0,
                        4,
                    ),
                },
                {
                    "CategoryID": category_id,
                    "CategoryName": category_name,
                    "FeatureName": "UserPriorCategoryRideCount",
                    "FeatureRole": "category_count",
                    "available_rows": user_history_model_stats[category_id]["prior_category_rows"],
                    "available_pct": round(
                        (
                            user_history_model_stats[category_id]["prior_category_rows"]
                            / row_count
                            * 100
                        )
                        if row_count
                        else 0,
                        4,
                    ),
                },
                {
                    "CategoryID": category_id,
                    "CategoryName": category_name,
                    "FeatureName": "UserPriorCategoryPriceMean",
                    "FeatureRole": "category_value",
                    "available_rows": user_history_model_stats[category_id]["prior_category_mean_rows"],
                    "available_pct": round(
                        (
                            user_history_model_stats[category_id]["prior_category_mean_rows"]
                            / row_count
                            * 100
                        )
                        if row_count
                        else 0,
                        4,
                    ),
                },
            ]
        )

    overall_distribution_df = build_overall_distribution_df(
        overall_counter,
        metrics["rows_written"],
    )
    category_distribution_df = build_category_distribution_df(
        category_counter,
        target_category_totals,
    )
    cross_coverage_df = build_cross_feature_coverage_df(cross_coverage_rows)
    user_history_coverage_df = build_user_history_coverage_df(
        user_history_coverage_rows
    )
    write_validation_report(
        metrics,
        overall_distribution_df,
        category_distribution_df,
        cross_coverage_df,
        user_history_coverage_df,
    )

    log.info("=" * 60)
    log.info("Camada de features gerada com sucesso")
    log.info("=" * 60)
    return metrics


if __name__ == "__main__":
    main()
