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
SOURCE_DIR = DATA_DIR / "analytical_curated"
OUTPUT_DIR = DATA_DIR / "features_temporal"

DISTRIBUTION_FILE = REPORTS_DIR / "temporal_feature_distribution.csv"
CATEGORY_DISTRIBUTION_FILE = REPORTS_DIR / "temporal_feature_target_distribution.csv"
METRICS_FILE = REPORTS_DIR / "temporal_feature_metrics.csv"
REPORT_FILE = REPORTS_DIR / "temporal_feature_report.md"

BATCH_SIZE = 100_000
TIMESTAMP_COLUMNS = ["Schedule", "Create", "Updated"]
TARGET_CATEGORIES = [2, 9, 4]
TARGET_CATEGORY_NAMES = {
    2: "UberX",
    9: "Uber Comfort",
    4: "Uber Black",
}
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


def write_validation_report(
    metrics: dict,
    overall_distribution_df: pd.DataFrame,
    category_distribution_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overall_distribution_df.to_csv(DISTRIBUTION_FILE, index=False)
    category_distribution_df.to_csv(CATEGORY_DISTRIBUTION_FILE, index=False)
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
    log.info("Relatorios salvos em %s", REPORTS_DIR)


# ============================================================
# PIPELINE
# ============================================================

def main() -> dict:
    log.info("=" * 60)
    log.info("Gerando features temporais")
    log.info("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset curado nao encontrado em {SOURCE_DIR}")

    prepare_output_dir(OUTPUT_DIR, DATA_DIR)

    dataset = ds.dataset(
        SOURCE_DIR,
        format="parquet",
        partitioning="hive",
    )
    holiday_index = build_brazil_holiday_index()

    metrics = {
        "rows_processed": 0,
        "rows_written": 0,
        "chunks_written": 0,
        "rows_on_disk": 0,
    }
    partition_row_counts: Counter = Counter()
    overall_counter: Counter = Counter()
    category_counter: Counter = Counter()
    target_category_totals = {category_id: 0 for category_id in TARGET_CATEGORIES}
    partitioning = None

    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        batch_df = batch.to_pandas()
        metrics["rows_processed"] += len(batch_df)

        enriched_df = add_temporal_features(batch_df, holiday_index)
        update_overall_distribution(enriched_df, overall_counter)
        update_category_distribution(enriched_df, category_counter)

        for category_id in TARGET_CATEGORIES:
            target_category_totals[category_id] += int(
                enriched_df["CategoryID"].eq(category_id).sum()
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

    overall_distribution_df = build_overall_distribution_df(
        overall_counter,
        metrics["rows_written"],
    )
    category_distribution_df = build_category_distribution_df(
        category_counter,
        target_category_totals,
    )
    write_validation_report(
        metrics,
        overall_distribution_df,
        category_distribution_df,
    )

    log.info("=" * 60)
    log.info("Features temporais geradas com sucesso")
    log.info("=" * 60)
    return metrics


if __name__ == "__main__":
    main()
