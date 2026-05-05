import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from utils import (
    cap_upper_by_group,
    cast_object_columns_to_string,
    coerce_numeric_columns,
    drop_duplicate_keys,
    parse_datetime_columns,
    prepare_output_dir,
)

# ============================================================
# CONFIGURACAO
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
SOURCE_DIR = DATA_DIR / "analytical"
OUTPUT_DIR = DATA_DIR / "analytical_curated"

STRATEGY_REPORT_FILE = REPORTS_DIR / "data_treatment_strategy.md"
FIELD_STRATEGIES_FILE = REPORTS_DIR / "data_treatment_field_strategies.csv"
THRESHOLDS_FILE = REPORTS_DIR / "data_treatment_thresholds.csv"
RUN_METRICS_FILE = REPORTS_DIR / "data_treatment_run_metrics.csv"
METRICS_JSON_FILE = REPORTS_DIR / "data_treatment_metrics.json"

BATCH_SIZE = 100_000
TARGET_CAP_QUANTILE = 0.995
WAITING_TIME_CAP_QUANTILE = 0.99
MAX_SCHEDULE_CREATE_GAP_SECONDS = 300
FARE_ID_SENTINEL = "MISSING_FAREID"

LEAKAGE_COLUMNS = [
    "RidePrice",
    "Selected",
    "RideReasonSelectedEstimativeID",
]

SPARSE_OR_CONSTANT_COLUMNS = [
    "Car",
    "ProviderID",
    "RideProviderID",
    "RideCategoryID",
    "RideDriverLocationID",
    "ScheduledRide",
]

DATETIME_COLUMNS = ["Schedule", "Create", "Updated"]
COORDINATE_COLUMNS = [
    "OriginLat",
    "OriginLng",
    "DestinationLat",
    "DestinationLng",
]
NUMERIC_COLUMNS = [
    "RideEstimativeID",
    "RideID",
    "WaitingTime",
    "Price",
    "Fee",
    "RidePrice",
    "RideCategoryID",
    "CompanyID",
    "ProviderID",
    "RideProviderID",
    "TotalUsers",
    "RideDriverLocationID",
    "ScheduledRide",
    "ProductProviderID",
    "CategoryID",
    *COORDINATE_COLUMNS,
]
NUMERIC_IMPUTATION_COLUMNS = ["WaitingTime", "Fee", "TotalUsers"]


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
# METADADOS DAS ESTRATEGIAS
# ============================================================

FIELD_STRATEGIES = [
    {
        "field_scope": "FareID",
        "issue": "40.91% de nulos em rideestimative.",
        "strategy": "Imputar com sentinel categorico.",
        "implementation": (
            "Preencher nulos com MISSING_FAREID e criar a flag FareIDWasImputed."
        ),
        "reason": (
            "FareID e um identificador categorico; a imputacao explicita preserva "
            "missingness como sinal sem distorcer a moda."
        ),
    },
    {
        "field_scope": "WaitingTime, Fee, TotalUsers",
        "issue": "Campos numericos retidos podem receber nulos futuros.",
        "strategy": "Fallback de mediana.",
        "implementation": (
            "Aplicar mediana global apenas se surgirem nulos; nesta execucao nao houve imputacoes."
        ),
        "reason": "A mediana e robusta e nao puxa a distribuicao por caudas extremas.",
    },
    {
        "field_scope": "Price",
        "issue": "Assimetria forte e outliers extremos por categoria.",
        "strategy": "Remover Price <= 0 e aplicar capping superior por CategoryID.",
        "implementation": (
            "Remover registros com Price <= 0 e capar o restante no percentil 99.5 por CategoryID."
        ),
        "reason": (
            "O target tem comportamento diferente por categoria; o capping segmentado "
            "preserva volume e reduz distorcao nos extremos."
        ),
    },
    {
        "field_scope": "WaitingTime",
        "issue": "Cauda longa com outliers detectados no profiling inicial.",
        "strategy": "Capping superior global.",
        "implementation": (
            "Capar no percentil 99 global e registrar a flag WaitingTimeWasCapped."
        ),
        "reason": (
            "A feature e operacional e monotona; o clipping leve reduz extremos "
            "sem descartar observacoes."
        ),
    },
    {
        "field_scope": "RidePrice, Selected, RideReasonSelectedEstimativeID",
        "issue": "Vazamento e informacao pos-evento.",
        "strategy": "Remocao de colunas.",
        "implementation": "Excluir do dataset curado.",
        "reason": "Nao estao disponiveis de forma honesta no momento de inferencia.",
    },
    {
        "field_scope": (
            "Car, ProviderID, RideProviderID, RideCategoryID, RideDriverLocationID, ScheduledRide"
        ),
        "issue": "Esparsidade extrema ou coluna constante.",
        "strategy": "Remocao de colunas.",
        "implementation": "Excluir do dataset curado.",
        "reason": (
            "Mais de 99% de nulos ou ausencia de variancia tornam essas colunas "
            "custosas e pouco informativas."
        ),
    },
    {
        "field_scope": "RideEstimativeID",
        "issue": "Risco de duplicidade de chave de negocio.",
        "strategy": "Remocao de duplicatas por chave primaria.",
        "implementation": "Manter a primeira ocorrencia de cada RideEstimativeID.",
        "reason": "Garante 1 linha por estimativa e evita inflar treino/validacao.",
    },
    {
        "field_scope": "Create, Schedule, Updated",
        "issue": "Necessidade de integridade temporal para TSCV e auditoria.",
        "strategy": "Padronizacao datetime e remocao de cronologias impossiveis.",
        "implementation": (
            "Converter para datetime, remover registros com Create/Schedule invalidos, "
            "Updated < Create ou gap absoluto > 5 min entre Schedule e Create em corridas on-demand."
        ),
        "reason": (
            "Create e a ancora temporal confiavel; cronologias impossiveis devem sair do dataset."
        ),
    },
    {
        "field_scope": "OriginLat, OriginLng, DestinationLat, DestinationLng",
        "issue": "Campos espaciais obrigatorios para FE geografica.",
        "strategy": "Remocao de linhas invalidas.",
        "implementation": "Descartar linhas com coordenadas ausentes ou nao numericas.",
        "reason": "Distancia e zona geografica dependem de coordenadas validas.",
    },
]


# ============================================================
# LEITURA E THRESHOLDS
# ============================================================

def load_source_dataset() -> ds.Dataset:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset analitico nao encontrado em {SOURCE_DIR}"
        )

    return ds.dataset(
        SOURCE_DIR,
        format="parquet",
        partitioning="hive",
    )


def compute_reference_statistics(dataset: ds.Dataset) -> dict:
    log.info("Calculando thresholds e estatisticas de referencia...")
    reference_table = dataset.to_table(
        columns=["CategoryID", "Price", "WaitingTime", "Fee", "TotalUsers"]
    )
    reference_df = reference_table.to_pandas()
    reference_df = coerce_numeric_columns(
        reference_df,
        ["CategoryID", "Price", "WaitingTime", "Fee", "TotalUsers"],
    )

    valid_price_df = reference_df[reference_df["Price"] > 0].copy()
    price_cap_by_category = {
        int(category_id): float(cap_value)
        for category_id, cap_value in (
            valid_price_df.groupby("CategoryID")["Price"]
            .quantile(TARGET_CAP_QUANTILE)
            .items()
        )
    }

    numeric_medians = {
        column: float(valid_price_df[column].median())
        for column in NUMERIC_IMPUTATION_COLUMNS
    }
    waiting_time_upper = float(
        valid_price_df["WaitingTime"].quantile(WAITING_TIME_CAP_QUANTILE)
    )

    log.info(
        "Thresholds prontos | categorias=%s | waiting_time_p99=%.4f",
        len(price_cap_by_category),
        waiting_time_upper,
    )
    return {
        "price_cap_by_category": price_cap_by_category,
        "numeric_medians": numeric_medians,
        "waiting_time_upper": waiting_time_upper,
    }


# ============================================================
# TRANSFORMACOES
# ============================================================

def apply_treatments_to_batch(
    batch_df: pd.DataFrame,
    reference_stats: dict,
    seen_estimative_ids: set[int],
) -> tuple[pd.DataFrame, dict, Counter, Counter]:
    metrics = Counter()
    capped_by_category = Counter()
    written_by_category = Counter()

    batch_df = coerce_numeric_columns(batch_df, NUMERIC_COLUMNS)
    batch_df = parse_datetime_columns(batch_df, DATETIME_COLUMNS)

    batch_df, duplicate_key_rows = drop_duplicate_keys(
        batch_df,
        "RideEstimativeID",
        seen_estimative_ids,
    )
    metrics["duplicate_rideestimative_rows_removed"] += duplicate_key_rows

    rows_before_exact_dedup = len(batch_df)
    batch_df = batch_df.drop_duplicates()
    metrics["exact_duplicate_rows_removed"] += (
        rows_before_exact_dedup - len(batch_df)
    )

    invalid_create_or_schedule = (
        batch_df["Create"].isna() | batch_df["Schedule"].isna()
    )
    impossible_updated = (
        batch_df["Updated"].notna() & (batch_df["Updated"] < batch_df["Create"])
    )
    on_demand_schedule_gap = (
        batch_df["ScheduledRide"].fillna(0).eq(0)
        & (
            (batch_df["Schedule"] - batch_df["Create"])
            .dt.total_seconds()
            .abs()
            > MAX_SCHEDULE_CREATE_GAP_SECONDS
        )
    )

    metrics["invalid_create_or_schedule_rows_removed"] += int(
        invalid_create_or_schedule.sum()
    )
    metrics["updated_before_create_rows_removed"] += int(
        impossible_updated.sum()
    )
    metrics["schedule_create_gap_gt_5m_rows_removed"] += int(
        on_demand_schedule_gap.sum()
    )

    temporal_drop_mask = (
        invalid_create_or_schedule
        | impossible_updated
        | on_demand_schedule_gap
    )
    batch_df = batch_df.loc[~temporal_drop_mask].copy()

    invalid_coordinates = batch_df[COORDINATE_COLUMNS].isna().any(axis=1)
    metrics["missing_coordinate_rows_removed"] += int(invalid_coordinates.sum())
    batch_df = batch_df.loc[~invalid_coordinates].copy()

    nonpositive_price_mask = batch_df["Price"] <= 0
    metrics["nonpositive_price_rows_removed"] += int(nonpositive_price_mask.sum())
    batch_df = batch_df.loc[~nonpositive_price_mask].copy()

    for column, median_value in reference_stats["numeric_medians"].items():
        null_mask = batch_df[column].isna()
        null_count = int(null_mask.sum())
        metrics[f"{column.lower()}_median_imputations"] += null_count
        if null_count:
            batch_df.loc[null_mask, column] = median_value

    fare_missing_mask = (
        batch_df["FareID"].isna()
        | batch_df["FareID"].astype("string").str.strip().eq("")
    )
    metrics["fareid_sentinel_imputations"] += int(fare_missing_mask.sum())
    batch_df["FareID"] = batch_df["FareID"].astype("string")
    batch_df.loc[fare_missing_mask, "FareID"] = FARE_ID_SENTINEL
    batch_df["FareIDWasImputed"] = fare_missing_mask.astype("bool")

    waiting_time_upper = reference_stats["waiting_time_upper"]
    waiting_time_cap_mask = batch_df["WaitingTime"] > waiting_time_upper
    metrics["waiting_time_rows_capped"] += int(waiting_time_cap_mask.sum())
    batch_df["WaitingTimeWasCapped"] = waiting_time_cap_mask.astype("bool")
    batch_df.loc[waiting_time_cap_mask, "WaitingTime"] = waiting_time_upper

    batch_df, price_cap_mask = cap_upper_by_group(
        batch_df,
        value_column="Price",
        group_column="CategoryID",
        upper_bounds=reference_stats["price_cap_by_category"],
    )
    metrics["price_rows_capped"] += int(price_cap_mask.sum())
    batch_df["PriceWasCapped"] = price_cap_mask.astype("bool")
    capped_by_category.update(
        batch_df.loc[price_cap_mask, "CategoryID"]
        .astype("int64")
        .astype("string")
        .value_counts()
        .to_dict()
    )

    columns_to_drop = [
        column
        for column in (LEAKAGE_COLUMNS + SPARSE_OR_CONSTANT_COLUMNS)
        if column in batch_df.columns
    ]
    batch_df = batch_df.drop(columns=columns_to_drop)

    batch_df["CategoryID"] = batch_df["CategoryID"].astype("int64")
    batch_df = cast_object_columns_to_string(batch_df)

    written_by_category.update(
        batch_df["CategoryID"].astype("string").value_counts().to_dict()
    )
    return batch_df, dict(metrics), capped_by_category, written_by_category


# ============================================================
# ESCRITA
# ============================================================

def write_curated_dataset(
    dataset: ds.Dataset,
    reference_stats: dict,
) -> dict:
    prepare_output_dir(OUTPUT_DIR, DATA_DIR)

    summary = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "source_rows": 0,
        "rows_written": 0,
        "chunks_written": 0,
        "partition_row_counts": Counter(),
        "price_rows_capped_by_category": Counter(),
        "thresholds": {
            "price_cap_quantile": TARGET_CAP_QUANTILE,
            "waiting_time_cap_quantile": WAITING_TIME_CAP_QUANTILE,
            "waiting_time_upper": reference_stats["waiting_time_upper"],
            "price_cap_by_category": reference_stats["price_cap_by_category"],
        },
    }

    for metric_name in [
        "duplicate_rideestimative_rows_removed",
        "exact_duplicate_rows_removed",
        "invalid_create_or_schedule_rows_removed",
        "updated_before_create_rows_removed",
        "schedule_create_gap_gt_5m_rows_removed",
        "missing_coordinate_rows_removed",
        "nonpositive_price_rows_removed",
        "waitingtime_median_imputations",
        "fee_median_imputations",
        "totalusers_median_imputations",
        "fareid_sentinel_imputations",
        "waiting_time_rows_capped",
        "price_rows_capped",
    ]:
        summary[metric_name] = 0

    partitioning = None
    seen_estimative_ids: set[int] = set()

    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        batch_df = batch.to_pandas()
        summary["source_rows"] += len(batch_df)

        treated_df, metrics, capped_by_category, written_by_category = (
            apply_treatments_to_batch(
                batch_df=batch_df,
                reference_stats=reference_stats,
                seen_estimative_ids=seen_estimative_ids,
            )
        )

        for metric_name, metric_value in metrics.items():
            summary[metric_name] += metric_value

        summary["price_rows_capped_by_category"].update(capped_by_category)
        summary["partition_row_counts"].update(written_by_category)

        if treated_df.empty:
            continue

        table = pa.Table.from_pandas(treated_df, preserve_index=False)
        if partitioning is None:
            partitioning = ds.partitioning(
                pa.schema([("CategoryID", table.schema.field("CategoryID").type)]),
                flavor="hive",
            )

        summary["chunks_written"] += 1
        ds.write_dataset(
            table,
            base_dir=OUTPUT_DIR,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"part-{summary['chunks_written']:05d}-{{i}}.parquet",
        )
        summary["rows_written"] += len(treated_df)

        log.info(
            "Chunk curado #%s salvo | linhas=%s | acumulado=%s",
            summary["chunks_written"],
            len(treated_df),
            summary["rows_written"],
        )

    curated_dataset = ds.dataset(
        OUTPUT_DIR,
        format="parquet",
        partitioning="hive",
    )
    summary["rows_on_disk"] = curated_dataset.count_rows()
    summary["rows_removed_total"] = summary["source_rows"] - summary["rows_written"]
    summary["retention_pct"] = round(
        (summary["rows_written"] / summary["source_rows"]) * 100,
        4,
    )

    if summary["rows_on_disk"] != summary["rows_written"]:
        raise ValueError(
            "Volume em disco diferente do processado. "
            f"Disco={summary['rows_on_disk']} | Processado={summary['rows_written']}"
        )

    summary["partition_row_counts"] = dict(
        sorted(summary["partition_row_counts"].items(), key=lambda item: int(item[0]))
    )
    summary["price_rows_capped_by_category"] = dict(
        sorted(
            summary["price_rows_capped_by_category"].items(),
            key=lambda item: int(item[0]),
        )
    )
    return summary


# ============================================================
# RELATORIOS
# ============================================================

def build_thresholds_dataframe(reference_stats: dict) -> pd.DataFrame:
    rows = [
        {
            "feature": "WaitingTime",
            "group_key": "ALL",
            "quantile": WAITING_TIME_CAP_QUANTILE,
            "upper_cap": round(reference_stats["waiting_time_upper"], 4),
        }
    ]
    rows.extend(
        {
            "feature": "Price",
            "group_key": str(category_id),
            "quantile": TARGET_CAP_QUANTILE,
            "upper_cap": round(upper_cap, 4),
        }
        for category_id, upper_cap in sorted(
            reference_stats["price_cap_by_category"].items()
        )
    )
    return pd.DataFrame(rows)


def build_run_metrics_dataframe(summary: dict) -> pd.DataFrame:
    metrics_rows = [
        {
            "metric": "source_rows",
            "value": summary["source_rows"],
        },
        {
            "metric": "rows_written",
            "value": summary["rows_written"],
        },
        {
            "metric": "rows_removed_total",
            "value": summary["rows_removed_total"],
        },
        {
            "metric": "retention_pct",
            "value": summary["retention_pct"],
        },
        {
            "metric": "duplicate_rideestimative_rows_removed",
            "value": summary["duplicate_rideestimative_rows_removed"],
        },
        {
            "metric": "exact_duplicate_rows_removed",
            "value": summary["exact_duplicate_rows_removed"],
        },
        {
            "metric": "invalid_create_or_schedule_rows_removed",
            "value": summary["invalid_create_or_schedule_rows_removed"],
        },
        {
            "metric": "updated_before_create_rows_removed",
            "value": summary["updated_before_create_rows_removed"],
        },
        {
            "metric": "schedule_create_gap_gt_5m_rows_removed",
            "value": summary["schedule_create_gap_gt_5m_rows_removed"],
        },
        {
            "metric": "missing_coordinate_rows_removed",
            "value": summary["missing_coordinate_rows_removed"],
        },
        {
            "metric": "nonpositive_price_rows_removed",
            "value": summary["nonpositive_price_rows_removed"],
        },
        {
            "metric": "fareid_sentinel_imputations",
            "value": summary["fareid_sentinel_imputations"],
        },
        {
            "metric": "waitingtime_median_imputations",
            "value": summary["waitingtime_median_imputations"],
        },
        {
            "metric": "fee_median_imputations",
            "value": summary["fee_median_imputations"],
        },
        {
            "metric": "totalusers_median_imputations",
            "value": summary["totalusers_median_imputations"],
        },
        {
            "metric": "waiting_time_rows_capped",
            "value": summary["waiting_time_rows_capped"],
        },
        {
            "metric": "price_rows_capped",
            "value": summary["price_rows_capped"],
        },
    ]
    return pd.DataFrame(metrics_rows)


def write_reports(
    reference_stats: dict,
    summary: dict,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    strategy_df = pd.DataFrame(FIELD_STRATEGIES)
    thresholds_df = build_thresholds_dataframe(reference_stats)
    run_metrics_df = build_run_metrics_dataframe(summary)

    strategy_df.to_csv(FIELD_STRATEGIES_FILE, index=False)
    thresholds_df.to_csv(THRESHOLDS_FILE, index=False)
    run_metrics_df.to_csv(RUN_METRICS_FILE, index=False)
    METRICS_JSON_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "# Estrategias de Tratamento de Nulos, Outliers e Inconsistencias",
        "",
        f"- Gerado em: `{summary['generated_at']}`",
        f"- Fonte: `{SOURCE_DIR}`",
        f"- Saida curada: `{OUTPUT_DIR}`",
        f"- Linhas na origem: `{summary['source_rows']}`",
        f"- Linhas apos tratamento: `{summary['rows_written']}`",
        f"- Linhas removidas: `{summary['rows_removed_total']}`",
        f"- Retencao: `{summary['retention_pct']}%`",
        "",
        "## Regras Definidas",
        "",
        strategy_df.to_markdown(index=False),
        "",
        "## Thresholds de Capping",
        "",
        thresholds_df.to_markdown(index=False),
        "",
        "## Impacto da Execucao",
        "",
        run_metrics_df.to_markdown(index=False),
        "",
        "## Distribuicao das Particoes Curadas",
        "",
        pd.DataFrame(
            [
                {
                    "CategoryID": category_id,
                    "rows_written": row_count,
                    "price_rows_capped": summary["price_rows_capped_by_category"].get(
                        category_id,
                        0,
                    ),
                }
                for category_id, row_count in summary["partition_row_counts"].items()
            ]
        ).to_markdown(index=False),
        "",
        "## Interpretacao DS",
        "",
        (
            "- `FareID` ficou com imputacao categorica explicita porque o missing e volumoso "
            "e potencialmente informativo; forcar a moda esconderia esse padrao."
        ),
        (
            "- `Price` passou a usar capping superior no percentil 99.5 por `CategoryID`, "
            "o que respeita a heterogeneidade entre produtos e reduz a influencia dos extremos."
        ),
        (
            "- `WaitingTime` recebeu capping leve no percentil 99 global para reduzir cauda "
            "operacional sem descarte de volume."
        ),
        (
            "- Colunas com vazamento (`RidePrice`, `Selected`, `RideReasonSelectedEstimativeID`) "
            "e colunas ultra-esparsas/constantes foram removidas do dataset curado."
        ),
        (
            "- Registros temporalmente impossiveis, coordenadas invalidas e `Price <= 0` "
            "foram configurados para remocao por regra de negocio."
        ),
        "",
        "## Conclusao",
        "",
        (
            "- O pipeline agora produz uma camada `analytical_curated` pronta para EDA "
            "orientada a modelagem e para a proxima etapa de feature engineering."
        ),
        (
            "- O dataset bruto consolidado em `data/analytical` permanece preservado para "
            "auditoria e reproducibilidade."
        ),
    ]
    STRATEGY_REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    log.info("Relatorio salvo em %s", STRATEGY_REPORT_FILE)


# ============================================================
# MAIN
# ============================================================

def main() -> dict:
    log.info("=" * 60)
    log.info("Iniciando curadoria do dataset analitico")
    log.info("=" * 60)

    dataset = load_source_dataset()
    reference_stats = compute_reference_statistics(dataset)
    summary = write_curated_dataset(dataset, reference_stats)
    write_reports(reference_stats, summary)

    log.info("=" * 60)
    log.info("Curadoria finalizada com sucesso")
    log.info("=" * 60)
    return summary


if __name__ == "__main__":
    main()
