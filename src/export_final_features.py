from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from utils import prepare_output_dir


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
SOURCE_DIR = DATA_DIR / "features_temporal"
OUTPUT_DIR = DATA_DIR / "final_features"

SUMMARY_FILE = REPORTS_DIR / "final_feature_exports_summary.csv"
REPORT_FILE = REPORTS_DIR / "final_feature_exports.md"

TARGET_COLUMN = "Price"
TIME_COLUMN = "Create"
TIME_ORDER_COLUMN = "CreateDate"

CATEGORY_EXPORTS = {
    2: {
        "dataset_name": "UberX",
        "file_name": "features_uberx.parquet",
        "own_cross_column": "Price_UberX",
        "auxiliary_cross_columns": ["Price_Comfort", "Price_Black"],
    },
    9: {
        "dataset_name": "Uber Comfort",
        "file_name": "features_comfort.parquet",
        "own_cross_column": "Price_Comfort",
        "auxiliary_cross_columns": ["Price_UberX", "Price_Black"],
    },
    4: {
        "dataset_name": "Uber Black",
        "file_name": "features_black.parquet",
        "own_cross_column": "Price_Black",
        "auxiliary_cross_columns": ["Price_UberX", "Price_Comfort"],
    },
}

TEMPORAL_FEATURE_COLUMNS = [
    "ScheduleHour",
    "ScheduleDayOfWeek",
    "ScheduleDayOfWeekName",
    "ScheduleMonth",
    "ScheduleQuarter",
    "ScheduleIsHolidayBR",
    "ScheduleDayPeriod",
    "CreateHour",
    "CreateDayOfWeek",
    "CreateDayOfWeekName",
    "CreateMonth",
    "CreateQuarter",
    "CreateIsHolidayBR",
    "CreateDayPeriod",
]
USER_HISTORY_FEATURE_COLUMNS = [
    "UserPriorRideCount",
    "UserPriorPaidPriceMean",
    "UserPriorCategoryRideCount",
    "UserPriorCategoryPriceMean",
]
QUALITY_FEATURE_COLUMNS = [
    "FareIDWasImputed",
    "WaitingTimeWasCapped",
]
HIGH_LEAKAGE_COLUMNS = [
    "Updated",
    "UpdatedHour",
    "UpdatedDayOfWeek",
    "UpdatedDayOfWeekName",
    "UpdatedMonth",
    "UpdatedQuarter",
    "UpdatedIsHolidayBR",
    "UpdatedDayPeriod",
    "PriceWasCapped",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def get_export_columns(
    source_columns: list[str],
    own_cross_column: str,
) -> list[str]:
    excluded_columns = set(HIGH_LEAKAGE_COLUMNS + [own_cross_column])
    ordered_columns = [column for column in source_columns if column not in excluded_columns]
    return ordered_columns


def validate_source_schema(source_columns: list[str]) -> None:
    required_columns = {
        "RideEstimativeID",
        "RideID",
        "CategoryID",
        "ProductID",
        TARGET_COLUMN,
        TIME_COLUMN,
        "Schedule",
        *TEMPORAL_FEATURE_COLUMNS,
        *USER_HISTORY_FEATURE_COLUMNS,
        *QUALITY_FEATURE_COLUMNS,
        "Price_UberX",
        "Price_Comfort",
        "Price_Black",
    }
    missing_columns = sorted(required_columns.difference(source_columns))
    if missing_columns:
        raise ValueError(
            f"Dataset de origem sem as colunas esperadas: {missing_columns}"
        )


def build_expected_engineered_columns(
    export_spec: dict,
) -> list[str]:
    return (
        TEMPORAL_FEATURE_COLUMNS
        + export_spec["auxiliary_cross_columns"]
        + USER_HISTORY_FEATURE_COLUMNS
        + QUALITY_FEATURE_COLUMNS
        + [TIME_ORDER_COLUMN]
    )


def export_category_dataset(
    dataset: ds.Dataset,
    category_id: int,
    export_spec: dict,
    source_columns: list[str],
) -> dict:
    export_columns = get_export_columns(
        source_columns=source_columns,
        own_cross_column=export_spec["own_cross_column"],
    )
    filter_expression = ds.field("CategoryID") == category_id
    category_df = dataset.to_table(
        columns=export_columns,
        filter=filter_expression,
    ).to_pandas()
    if category_df.empty:
        raise ValueError(f"Nenhuma linha encontrada para CategoryID={category_id}.")

    category_df[TIME_COLUMN] = pd.to_datetime(category_df[TIME_COLUMN], errors="coerce")
    if category_df[TIME_COLUMN].isna().any():
        raise ValueError(
            f"Existem timestamps invalidos em {TIME_COLUMN} para CategoryID={category_id}."
        )

    category_df[TIME_ORDER_COLUMN] = category_df[TIME_COLUMN].dt.floor("D")
    category_df = category_df.sort_values(
        by=[TIME_ORDER_COLUMN, TIME_COLUMN, "RideID", "RideEstimativeID"],
        kind="stable",
    ).reset_index(drop=True)

    if category_df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Existem valores nulos em {TARGET_COLUMN} para CategoryID={category_id}."
        )

    expected_engineered_columns = build_expected_engineered_columns(export_spec)
    missing_engineered_columns = sorted(
        set(expected_engineered_columns).difference(category_df.columns)
    )
    if missing_engineered_columns:
        raise ValueError(
            "Colunas engenheiradas ausentes na exportacao "
            f"de CategoryID={category_id}: {missing_engineered_columns}"
        )

    if export_spec["own_cross_column"] in category_df.columns:
        raise ValueError(
            f"Leakage detectado: {export_spec['own_cross_column']} ainda presente "
            f"na exportacao de CategoryID={category_id}."
        )

    output_file = OUTPUT_DIR / export_spec["file_name"]
    output_table = pa.Table.from_pandas(category_df, preserve_index=False)
    pq.write_table(output_table, output_file, compression="snappy")

    written_dataset = ds.dataset(output_file, format="parquet")
    written_row_count = int(written_dataset.count_rows())
    if written_row_count != len(category_df):
        raise ValueError(
            "Volume inconsistente apos escrita em Parquet "
            f"para CategoryID={category_id}: df={len(category_df)} | parquet={written_row_count}"
        )

    return {
        "CategoryID": category_id,
        "DatasetName": export_spec["dataset_name"],
        "OutputFile": str(output_file),
        "rows_written": int(written_row_count),
        "column_count": int(len(category_df.columns)),
        "engineered_feature_count": int(len(expected_engineered_columns)),
        "target_present": TARGET_COLUMN in category_df.columns,
        "target_nulls": int(category_df[TARGET_COLUMN].isna().sum()),
        "time_column_present": TIME_COLUMN in category_df.columns,
        "time_order_column_present": TIME_ORDER_COLUMN in category_df.columns,
        "time_nulls": int(category_df[TIME_COLUMN].isna().sum()),
        "min_create": category_df[TIME_COLUMN].min().isoformat(),
        "max_create": category_df[TIME_COLUMN].max().isoformat(),
        "own_cross_removed": export_spec["own_cross_column"] not in category_df.columns,
        "aux_cross_1": export_spec["auxiliary_cross_columns"][0],
        "aux_cross_1_present": export_spec["auxiliary_cross_columns"][0] in category_df.columns,
        "aux_cross_2": export_spec["auxiliary_cross_columns"][1],
        "aux_cross_2_present": export_spec["auxiliary_cross_columns"][1] in category_df.columns,
        "target_positive_rows": int(category_df[TARGET_COLUMN].gt(0).sum()),
    }


def write_report(summary_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    report_lines = [
        "# Exportacao Final dos Datasets de Features por Categoria",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{SOURCE_DIR}`",
        f"- Saida: `{OUTPUT_DIR}`",
        "",
        "## Resultado DE",
        "",
        "- Foram gerados tres arquivos Parquet finais, um por categoria-alvo, prontos para modelagem e TSCV.",
        "- Cada arquivo preserva o target `Price`, a coluna temporal `Create` e a coluna derivada `CreateDate` para ordenacao.",
        "- Colunas de leakage alto identificadas na etapa 3.5 foram removidas da camada final (`Updated*`, `PriceWasCapped` e o preco cruzado da propria categoria).",
        "",
        "## Validacao DS",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Conclusao",
        "",
        "- Os volumes por categoria estao coerentes com a camada `features_temporal`.",
        "- O target `Price` esta presente em todos os arquivos, sem nulos.",
        "- As features engenheiradas esperadas foram mantidas, incluindo temporais, historico por usuario e precos cruzados auxiliares.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Exportando datasets finais por categoria")
    log.info("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset de features nao encontrado em {SOURCE_DIR}")

    prepare_output_dir(OUTPUT_DIR, DATA_DIR)
    dataset = ds.dataset(
        SOURCE_DIR,
        format="parquet",
        partitioning="hive",
    )
    source_columns = dataset.schema.names
    validate_source_schema(source_columns)

    summary_rows = []
    for category_id, export_spec in CATEGORY_EXPORTS.items():
        summary = export_category_dataset(
            dataset=dataset,
            category_id=category_id,
            export_spec=export_spec,
            source_columns=source_columns,
        )
        summary_rows.append(summary)
        log.info(
            "Arquivo final salvo | categoria=%s | linhas=%s | arquivo=%s",
            export_spec["dataset_name"],
            summary["rows_written"],
            export_spec["file_name"],
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="CategoryID",
        kind="stable",
    ).reset_index(drop=True)
    write_report(summary_df)

    log.info("=" * 60)
    log.info("Datasets finais exportados com sucesso")
    log.info("=" * 60)
    return summary_df


if __name__ == "__main__":
    main()
