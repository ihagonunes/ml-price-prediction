import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

# ============================================================
# CONFIGURACAO
# ============================================================

BASE_DIR = Path(__file__).parent.parent
ANALYTICAL_DIR = BASE_DIR / "data" / "analytical"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_REPORT_FILE = REPORTS_DIR / "parquet_profiling_metrics.json"
MARKDOWN_REPORT_FILE = REPORTS_DIR / "parquet_profiling_report.md"

HIGH_NULL_THRESHOLD = 30.0
HIGH_OUTLIER_THRESHOLD = 1.0
CATEGORICAL_MAX_UNIQUES = 1_000
CATEGORICAL_MAX_RATIO = 0.20

TABLE_CONFIGS = {
    "rideestimative": {
        "source_columns": [
            "RideEstimativeID",
            "RideID",
            "ProductID",
            "WaitingTime",
            "Price",
            "FareID",
            "Selected",
            "RideReasonSelectedEstimativeID",
            "Fee",
        ],
        "primary_key": "RideEstimativeID",
        "rename_map": {},
        "deduplicate_by_primary_key": False,
        "scope_note": (
            "Tabela fato reconstruida diretamente do Parquet analitico, "
            "sem necessidade de deduplicacao adicional."
        ),
    },
    "ride": {
        "source_columns": [
            "RideID",
            "UserID",
            "Schedule",
            "Create",
            "RideStatusID",
            "CompanyID",
            "ProviderID",
            "RideProviderID",
            "RidePrice",
            "Updated",
            "RideCategoryID",
            "TotalUsers",
            "Car",
            "RideDriverLocationID",
            "ScheduledRide",
        ],
        "primary_key": "RideID",
        "rename_map": {
            "RidePrice": "price",
            "RideCategoryID": "CategoryID",
        },
        "deduplicate_by_primary_key": True,
        "scope_note": (
            "Tabela reconstruida a partir do Parquet analitico. "
            "Contem apenas RideIDs com pelo menos uma estimativa associada."
        ),
    },
    "product": {
        "source_columns": [
            "ProductID",
            "ProductProviderID",
            "CategoryID",
            "ProductDescription",
        ],
        "primary_key": "ProductID",
        "rename_map": {
            "ProductProviderID": "ProviderID",
            "ProductDescription": "Description",
        },
        "deduplicate_by_primary_key": True,
        "scope_note": (
            "Tabela reconstruida a partir do Parquet analitico. "
            "Contem apenas ProductIDs efetivamente usados no dataset consolidado."
        ),
    },
}


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
# LEITURA E RECONSTRUCAO LOGICA
# ============================================================

def ensure_analytical_dataset_exists() -> None:
    if not ANALYTICAL_DIR.exists():
        raise FileNotFoundError(
            f"Dataset analitico nao encontrado em {ANALYTICAL_DIR}. "
            "Execute primeiro o pipeline de ingestion."
        )


def load_table_from_parquet(table_name: str, config: dict) -> tuple[pd.DataFrame, dict]:
    log.info("Carregando visao logica de %s a partir do Parquet...", table_name)

    dataset = ds.dataset(ANALYTICAL_DIR, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=config["source_columns"])
    df = table.to_pandas().convert_dtypes(dtype_backend="pyarrow")

    loaded_rows = len(df)
    full_unique_df = df.drop_duplicates(ignore_index=True)
    distinct_rows = len(full_unique_df)

    if config["deduplicate_by_primary_key"]:
        logical_df = full_unique_df.drop_duplicates(
            subset=[config["primary_key"]],
            keep="first",
            ignore_index=True,
        )
    else:
        logical_df = full_unique_df

    conflicting_primary_keys = int(
        full_unique_df[config["primary_key"]].value_counts(dropna=False).gt(1).sum()
    )

    reconstruction_stats = {
        "rows_scanned_from_parquet": loaded_rows,
        "distinct_rows_before_pk_dedup": distinct_rows,
        "logical_rows_after_pk_dedup": len(logical_df),
        "exact_duplicate_rows_in_parquet_view": loaded_rows - distinct_rows,
        "conflicting_primary_keys": conflicting_primary_keys,
    }

    if config["rename_map"]:
        logical_df = logical_df.rename(columns=config["rename_map"])

    log.info(
        "%s | parquet_rows=%s | logical_rows=%s | exact_duplicates=%s | conflicting_keys=%s",
        table_name,
        reconstruction_stats["rows_scanned_from_parquet"],
        reconstruction_stats["logical_rows_after_pk_dedup"],
        reconstruction_stats["exact_duplicate_rows_in_parquet_view"],
        reconstruction_stats["conflicting_primary_keys"],
    )

    return logical_df, reconstruction_stats


# ============================================================
# METRICAS DE QUALIDADE
# ============================================================

def serialize_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "as_py"):
        return value.as_py()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def summarize_nulls(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    row_count = len(df)
    null_counts = df.isna().sum()

    null_summary = {}
    for column in df.columns:
        pct = round((float(null_counts[column]) / row_count) * 100, 2) if row_count else 0.0
        null_summary[column] = {
            "null_count": int(null_counts[column]),
            "null_pct": pct,
        }

    high_null_columns = [
        {
            "column": column,
            "null_pct": stats["null_pct"],
            "null_count": stats["null_count"],
        }
        for column, stats in null_summary.items()
        if stats["null_pct"] >= HIGH_NULL_THRESHOLD
    ]

    high_null_columns.sort(key=lambda item: item["null_pct"], reverse=True)
    return null_summary, high_null_columns


def summarize_duplicates(
    df: pd.DataFrame,
    primary_key: str,
    reconstruction_stats: dict,
) -> dict:
    duplicate_primary_keys = int(df.duplicated(subset=[primary_key]).sum())
    duplicate_full_rows = int(df.duplicated().sum())

    return {
        "primary_key": primary_key,
        "duplicate_primary_keys_in_logical_table": duplicate_primary_keys,
        "duplicate_full_rows_in_logical_table": duplicate_full_rows,
        "exact_duplicate_rows_in_parquet_view": reconstruction_stats[
            "exact_duplicate_rows_in_parquet_view"
        ],
        "conflicting_primary_keys": reconstruction_stats["conflicting_primary_keys"],
    }


def summarize_dtypes(df: pd.DataFrame) -> dict:
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def identify_categorical_columns(df: pd.DataFrame, primary_key: str) -> tuple[list[str], dict]:
    categorical_columns = []
    high_cardinality_columns = {}
    row_count = len(df)

    for column in df.columns:
        if column == primary_key:
            continue

        non_null = df[column].dropna()
        unique_count = int(non_null.nunique())
        unique_ratio = round(unique_count / len(non_null), 4) if len(non_null) else 0.0
        is_string = pd.api.types.is_string_dtype(df[column])
        is_small_numeric_domain = (
            pd.api.types.is_numeric_dtype(df[column]) and unique_count <= 50
        )

        if is_string and (unique_count > CATEGORICAL_MAX_UNIQUES or unique_ratio > CATEGORICAL_MAX_RATIO):
            high_cardinality_columns[column] = {
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
            }
            continue

        if is_string or is_small_numeric_domain:
            categorical_columns.append(column)

    return categorical_columns, high_cardinality_columns


def summarize_categorical_columns(df: pd.DataFrame, primary_key: str) -> tuple[dict, dict]:
    categorical_columns, high_cardinality_columns = identify_categorical_columns(
        df,
        primary_key,
    )

    categorical_summary = {}
    for column in categorical_columns:
        non_null = df[column].dropna()
        counts = non_null.value_counts(dropna=False).head(10)
        categorical_summary[column] = {
            "unique_count": int(non_null.nunique()),
            "unique_ratio": round(
                int(non_null.nunique()) / len(non_null),
                4,
            ) if len(non_null) else 0.0,
            "top_values": {
                str(index): int(value)
                for index, value in counts.items()
            },
        }

    return categorical_summary, high_cardinality_columns


def summarize_constant_columns(df: pd.DataFrame) -> list[str]:
    constant_columns = []
    for column in df.columns:
        if int(df[column].nunique(dropna=False)) <= 1:
            constant_columns.append(column)
    return constant_columns


def is_outlier_candidate(column: str, series: pd.Series, primary_key: str) -> bool:
    if column == primary_key:
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if column.lower().endswith("id"):
        return False

    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if len(non_null) == 0:
        return False
    if non_null.nunique() <= 10:
        return False

    return True


def summarize_outliers(df: pd.DataFrame, primary_key: str) -> dict:
    outlier_summary = {}

    for column in df.columns:
        if not is_outlier_candidate(column, df[column], primary_key):
            continue

        numeric_series = pd.to_numeric(df[column], errors="coerce").dropna()
        q1 = float(numeric_series.quantile(0.25))
        q3 = float(numeric_series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            lower_bound = q1
            upper_bound = q3
            outlier_mask = pd.Series(False, index=numeric_series.index)
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)

        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(numeric_series)) * 100, 2) if len(numeric_series) else 0.0

        outlier_summary[column] = {
            "min": round(float(numeric_series.min()), 4),
            "q1": round(q1, 4),
            "median": round(float(numeric_series.median()), 4),
            "q3": round(q3, 4),
            "max": round(float(numeric_series.max()), 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
        }

    return outlier_summary


def build_table_profile(table_name: str, config: dict) -> dict:
    logical_df, reconstruction_stats = load_table_from_parquet(table_name, config)

    null_summary, high_null_columns = summarize_nulls(logical_df)
    duplicates_summary = summarize_duplicates(
        logical_df,
        config["rename_map"].get(config["primary_key"], config["primary_key"]),
        reconstruction_stats,
    )
    dtypes_summary = summarize_dtypes(logical_df)
    categorical_summary, high_cardinality_columns = summarize_categorical_columns(
        logical_df,
        config["rename_map"].get(config["primary_key"], config["primary_key"]),
    )
    constant_columns = summarize_constant_columns(logical_df)
    outlier_summary = summarize_outliers(
        logical_df,
        config["rename_map"].get(config["primary_key"], config["primary_key"]),
    )

    high_outlier_columns = [
        {
            "column": column,
            "outlier_pct": stats["outlier_pct"],
            "outlier_count": stats["outlier_count"],
        }
        for column, stats in outlier_summary.items()
        if stats["outlier_pct"] >= HIGH_OUTLIER_THRESHOLD
    ]
    high_outlier_columns.sort(key=lambda item: item["outlier_pct"], reverse=True)

    profile = {
        "table_name": table_name,
        "scope_note": config["scope_note"],
        "row_count": len(logical_df),
        "column_count": len(logical_df.columns),
        "reconstruction": reconstruction_stats,
        "dtypes": dtypes_summary,
        "nulls": null_summary,
        "high_null_columns": high_null_columns,
        "duplicates": duplicates_summary,
        "categorical_cardinality": categorical_summary,
        "high_cardinality_columns": high_cardinality_columns,
        "constant_columns": constant_columns,
        "outliers": outlier_summary,
        "high_outlier_columns": high_outlier_columns,
    }

    log.info(
        "%s | high_null_columns=%s | high_outlier_columns=%s | high_cardinality_columns=%s",
        table_name,
        len(high_null_columns),
        len(high_outlier_columns),
        len(high_cardinality_columns),
    )
    return profile


# ============================================================
# RELATORIOS
# ============================================================

def render_findings_line(items: list[dict], pct_key: str) -> str:
    if not items:
        return "Nenhum achado relevante."
    return "; ".join(
        f"{item['column']} ({item[pct_key]}%)"
        for item in items[:8]
    )


def build_interpretation_lines(table_name: str, table_report: dict) -> list[str]:
    lines = []

    high_null_columns = table_report["high_null_columns"]
    if high_null_columns:
        critical_nulls = [item["column"] for item in high_null_columns if item["null_pct"] >= 90.0]
        moderate_nulls = [item["column"] for item in high_null_columns if item["null_pct"] < 90.0]

        if critical_nulls:
            lines.append(
                "Campos com nulos extremos e forte candidato a descarte ou uso muito restrito: "
                + ", ".join(critical_nulls)
                + "."
            )
        if moderate_nulls:
            lines.append(
                "Campos com nulos relevantes que pedem regra de imputacao ou analise de missingness: "
                + ", ".join(moderate_nulls)
                + "."
            )
    else:
        lines.append("Nao foram encontrados campos com nulos acima do limiar de alerta.")

    duplicates = table_report["duplicates"]
    if (
        duplicates["duplicate_primary_keys_in_logical_table"] == 0
        and duplicates["conflicting_primary_keys"] == 0
    ):
        lines.append("Nao ha duplicatas suspeitas na tabela logica reconstruida.")
    else:
        lines.append(
            "Existem duplicatas suspeitas ou chaves conflitantes que precisam de saneamento antes da modelagem."
        )

    if table_report["high_outlier_columns"]:
        outlier_parts = []
        for item in table_report["high_outlier_columns"][:3]:
            stats = table_report["outliers"][item["column"]]
            outlier_parts.append(
                f"{item['column']} ({item['outlier_pct']}% acima/abaixo de [{stats['lower_bound']}, {stats['upper_bound']}])"
            )
        lines.append(
            "Outliers iniciais detectados via IQR: " + "; ".join(outlier_parts) + "."
        )
    else:
        lines.append("Nao houve sinal forte de outliers nas variaveis numericas elegiveis.")

    if table_report["high_cardinality_columns"]:
        columns = ", ".join(table_report["high_cardinality_columns"].keys())
        lines.append(
            "Colunas de alta cardinalidade devem ser tratadas como identificadores/temporais ou passar por encoding especifico: "
            + columns
            + "."
        )

    if table_report["constant_columns"]:
        lines.append(
            "Colunas constantes e candidatas a remocao por nao agregarem variancia: "
            + ", ".join(table_report["constant_columns"])
            + "."
        )

    if table_name == "ride":
        lines.append(
            "Os campos Schedule, Create e Updated estao tipados como string no Parquet e devem ser convertidos para datetime antes do feature engineering."
        )

    return lines


def build_markdown_report(report: dict) -> str:
    lines = [
        "# Profiling do Dataset Analitico",
        "",
        f"- Gerado em: `{report['generated_at']}`",
        f"- Fonte: `{report['source']}`",
        f"- Escopo: profiling reconstruido a partir do Parquet analitico, sem releitura dos CSVs.",
        "",
        "## Resumo Executivo",
        "",
    ]

    for table_name, table_report in report["tables"].items():
        lines.extend(
            [
                f"### {table_name}",
                "",
                f"- Linhas logicas: `{table_report['row_count']}`",
                f"- Colunas: `{table_report['column_count']}`",
                f"- Nota de escopo: {table_report['scope_note']}",
                f"- Nulos relevantes: {render_findings_line(table_report['high_null_columns'], 'null_pct')}",
                f"- Duplicatas suspeitas: `PK duplicada={table_report['duplicates']['duplicate_primary_keys_in_logical_table']}` | "
                f"`PK conflitante={table_report['duplicates']['conflicting_primary_keys']}`",
                f"- Outliers relevantes: {render_findings_line(table_report['high_outlier_columns'], 'outlier_pct')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretacao DS",
            "",
        ]
    )

    for table_name, table_report in report["tables"].items():
        lines.extend(
            [
                f"### {table_name}",
                "",
            ]
        )
        for item in build_interpretation_lines(table_name, table_report):
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(
        [
            "## Detalhamento",
            "",
        ]
    )

    for table_name, table_report in report["tables"].items():
        lines.extend(
            [
                f"### {table_name}",
                "",
                f"- Reconstrucao: {table_report['reconstruction']}",
                f"- Tipos: {table_report['dtypes']}",
                f"- Campos com nulos >= {HIGH_NULL_THRESHOLD}%: {table_report['high_null_columns'] or 'nenhum'}",
                f"- Cardinalidade categorica: {table_report['categorical_cardinality'] or 'nenhuma'}",
                f"- Campos de alta cardinalidade: {table_report['high_cardinality_columns'] or 'nenhum'}",
                f"- Colunas constantes: {table_report['constant_columns'] or 'nenhuma'}",
                f"- Outliers (IQR): {table_report['high_outlier_columns'] or 'nenhum'}",
                "",
            ]
        )

    return "\n".join(lines)


def persist_reports(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    METRICS_REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=serialize_scalar),
        encoding="utf-8",
    )
    MARKDOWN_REPORT_FILE.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    log.info("Relatorio JSON salvo em %s", METRICS_REPORT_FILE)
    log.info("Relatorio Markdown salvo em %s", MARKDOWN_REPORT_FILE)


# ============================================================
# MAIN
# ============================================================

def main() -> dict:
    ensure_analytical_dataset_exists()

    report = {
        "generated_at": datetime.now().isoformat(),
        "source": str(ANALYTICAL_DIR),
        "tables": {},
    }

    for table_name, config in TABLE_CONFIGS.items():
        report["tables"][table_name] = build_table_profile(table_name, config)

    persist_reports(report)
    return report


if __name__ == "__main__":
    main()
