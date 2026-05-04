import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# ============================================================
# CONFIGURACAO
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR = DATA_DIR / "analytical"
VALIDATION_REPORT_FILE = REPORTS_DIR / "analytical_dataset_validation.json"

RIDE_FILE = DATA_DIR / "ride_v2.csv"
PRODUCT_FILE = DATA_DIR / "product.csv"
RIDE_EST_FILE = DATA_DIR / "rideestimative_v3.csv"
RIDE_ADDRESS_FILE = DATA_DIR / "rideaddress_v1.csv"

CHUNKSIZE = 100_000
CSV_OPTIONS = {
    "sep": ";",
    "encoding": "utf-8",
    "na_values": ["NULL"],
    "keep_default_na": True,
    "low_memory": False,
}

PII_FIELDS = [
    "Name",
    "Phone",
    "Driver",
    "Plate",
    "DriverPhone",
    "DriverPicture",
    "Registration",
]

ADDRESS_TYPE_PREFIX = {
    1: "Origin",
    2: "Destination",
}

RIDE_RENAME_MAP = {
    "price": "RidePrice",
    "CategoryID": "RideCategoryID",
}

PRODUCT_RENAME_MAP = {
    "ProviderID": "ProductProviderID",
    "Description": "ProductDescription",
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
# UTILITARIOS
# ============================================================

def ensure_file_exists(filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")


def load_csv_header(filepath: Path) -> list[str]:
    ensure_file_exists(filepath)
    header = pd.read_csv(filepath, nrows=0, **CSV_OPTIONS)
    return list(header.columns)


def remove_pii_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if col not in PII_FIELDS]


def normalize_product_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def rename_columns_with_prefix(
    df: pd.DataFrame,
    prefix: str,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    exclude = exclude or set()
    rename_map = {
        column: f"{prefix}{column}"
        for column in df.columns
        if column not in exclude
    }
    return df.rename(columns=rename_map)


def validate_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    duplicated = df[key].duplicated().sum()
    if duplicated:
        sample = df.loc[df[key].duplicated(keep=False), [key]].head(5)
        raise ValueError(
            f"{name} possui {duplicated} chaves duplicadas em {key}. "
            f"Amostra: {sample.to_dict(orient='records')}"
        )


def cast_object_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = list(df.select_dtypes(include=["object"]).columns)
    if not object_columns:
        return df

    converted = df.copy()
    for column in object_columns:
        converted[column] = converted[column].astype("string")
    return converted


def prepare_output_dir(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    data_dir = DATA_DIR.resolve()

    if output_dir != data_dir and data_dir not in output_dir.parents:
        raise ValueError(
            f"Diretorio de saida fora do escopo esperado: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_validation_report(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Relatorio de validacao salvo em %s", VALIDATION_REPORT_FILE)


def read_standard_csv(
    filepath: Path,
    name: str,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    log.info("Lendo %s...", name)
    columns = remove_pii_columns(load_csv_header(filepath))
    df = pd.read_csv(
        filepath,
        usecols=columns,
        **CSV_OPTIONS,
    )

    if "ProductID" in df.columns:
        df["ProductID"] = normalize_product_id(df["ProductID"])

    if rename_map:
        df = df.rename(columns=rename_map)

    log.info("%s carregado | shape=%s", name, df.shape)
    return df


# ============================================================
# PERFIL BASE DE RIDEESTIMATIVE
# ============================================================

def profile_rideestimative(filepath: Path) -> dict:
    log.info("=" * 60)
    log.info("Iniciando perfil base de rideestimative_v3.csv")
    log.info("=" * 60)

    columns = load_csv_header(filepath)
    log.info("Schema detectado (%s colunas): %s", len(columns), columns)

    stats = {
        "rows": 0,
        "chunks": 0,
        "null_ride_id_rows": 0,
        "null_product_id_rows": 0,
        "active_ride_ids": set(),
        "active_product_ids": set(),
    }

    for chunk in pd.read_csv(
        filepath,
        usecols=["RideID", "ProductID"],
        chunksize=CHUNKSIZE,
        **CSV_OPTIONS,
    ):
        stats["chunks"] += 1
        stats["rows"] += len(chunk)

        chunk["ProductID"] = normalize_product_id(chunk["ProductID"])

        stats["null_ride_id_rows"] += int(chunk["RideID"].isna().sum())
        stats["null_product_id_rows"] += int(chunk["ProductID"].isna().sum())
        stats["active_ride_ids"].update(
            chunk["RideID"].dropna().astype("int64").tolist()
        )
        stats["active_product_ids"].update(
            chunk["ProductID"].dropna().tolist()
        )

        log.info(
            "Chunk base #%s | linhas=%s | acumulado=%s",
            stats["chunks"],
            len(chunk),
            stats["rows"],
        )

    if stats["null_ride_id_rows"] or stats["null_product_id_rows"]:
        raise ValueError(
            "rideestimative possui chaves nulas. "
            f"RideID nulos={stats['null_ride_id_rows']} | "
            f"ProductID nulos={stats['null_product_id_rows']}"
        )

    log.info(
        "Perfil base concluido | linhas=%s | ride_ids ativos=%s | product_ids ativos=%s",
        stats["rows"],
        len(stats["active_ride_ids"]),
        len(stats["active_product_ids"]),
    )
    return stats


# ============================================================
# DIMENSOES
# ============================================================

def prepare_ride_dimension(active_ride_ids: set[int]) -> pd.DataFrame:
    ride_df = read_standard_csv(
        RIDE_FILE,
        "ride_v2.csv",
        rename_map=RIDE_RENAME_MAP,
    )

    ride_df = ride_df[ride_df["RideID"].isin(active_ride_ids)].copy()
    validate_unique_key(ride_df, "RideID", "ride")

    missing_ride_ids = active_ride_ids - set(ride_df["RideID"].tolist())
    if missing_ride_ids:
        raise ValueError(
            f"ride nao cobre {len(missing_ride_ids)} RideIDs presentes em rideestimative."
        )

    ride_df["RideCategoryID"] = pd.to_numeric(
        ride_df["RideCategoryID"],
        errors="coerce",
    ).astype("Int64")

    log.info("ride pronto para join | shape=%s", ride_df.shape)
    return ride_df


def prepare_product_dimension(active_product_ids: set[str]) -> pd.DataFrame:
    product_df = read_standard_csv(
        PRODUCT_FILE,
        "product.csv",
        rename_map=PRODUCT_RENAME_MAP,
    )

    product_df = product_df[product_df["ProductID"].isin(active_product_ids)].copy()
    validate_unique_key(product_df, "ProductID", "product")

    missing_product_ids = active_product_ids - set(product_df["ProductID"].tolist())
    if missing_product_ids:
        sample = sorted(missing_product_ids)[:5]
        raise ValueError(
            f"product nao cobre {len(missing_product_ids)} ProductIDs ativos. "
            f"Amostra: {sample}"
        )

    product_df["CategoryID"] = pd.to_numeric(
        product_df["CategoryID"],
        errors="coerce",
    ).astype("Int64")

    if product_df["CategoryID"].isna().any():
        raise ValueError("Existem ProductIDs ativos com CategoryID nulo.")

    log.info("product pronto para join | shape=%s", product_df.shape)
    return product_df


def prepare_ride_address_dimension(active_ride_ids: set[int]) -> tuple[pd.DataFrame, dict]:
    address_df = read_standard_csv(RIDE_ADDRESS_FILE, "rideaddress_v1.csv")
    address_df = address_df[address_df["RideID"].isin(active_ride_ids)].copy()

    unexpected_types = sorted(
        set(address_df["RideAddressTypeID"].dropna().tolist())
        - set(ADDRESS_TYPE_PREFIX.keys())
    )
    if unexpected_types:
        raise ValueError(
            f"RideAddressTypeID inesperado encontrado: {unexpected_types}"
        )

    address_stats = {
        "active_ride_ids": len(active_ride_ids),
        "source_rows": len(address_df),
        "origin_rows": 0,
        "destination_rows": 0,
        "missing_origin_rows": 0,
        "missing_destination_rows": 0,
    }

    origin_df = address_df[address_df["RideAddressTypeID"] == 1].copy()
    destination_df = address_df[address_df["RideAddressTypeID"] == 2].copy()

    address_stats["origin_rows"] = len(origin_df)
    address_stats["destination_rows"] = len(destination_df)

    origin_df = origin_df.drop(columns=["RideAddressTypeID"])
    destination_df = destination_df.drop(columns=["RideAddressTypeID"])

    validate_unique_key(origin_df, "RideID", "rideaddress origem")
    validate_unique_key(destination_df, "RideID", "rideaddress destino")

    origin_df = rename_columns_with_prefix(origin_df, "Origin", {"RideID"})
    destination_df = rename_columns_with_prefix(destination_df, "Destination", {"RideID"})

    merged = origin_df.merge(
        destination_df,
        on="RideID",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    address_stats["missing_origin_rows"] = int((merged["_merge"] == "right_only").sum())
    address_stats["missing_destination_rows"] = int((merged["_merge"] == "left_only").sum())

    merged = merged.drop(columns=["_merge"])

    missing_ride_ids = active_ride_ids - set(merged["RideID"].tolist())
    if missing_ride_ids:
        raise ValueError(
            f"rideaddress nao cobre {len(missing_ride_ids)} RideIDs ativos."
        )

    log.info(
        "rideaddress consolidado | shape=%s | faltando origem=%s | faltando destino=%s",
        merged.shape,
        address_stats["missing_origin_rows"],
        address_stats["missing_destination_rows"],
    )
    return merged, address_stats


# ============================================================
# JOIN + PERSISTENCIA
# ============================================================

def merge_dimension(
    fact_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    join_key: str,
    dimension_name: str,
) -> pd.DataFrame:
    merged = fact_df.merge(
        dim_df,
        on=join_key,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing_rows = int((merged["_merge"] != "both").sum())
    if missing_rows:
        sample = (
            merged.loc[merged["_merge"] != "both", [join_key]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Join com {dimension_name} falhou para {missing_rows} linhas. "
            f"Amostra: {sample}"
        )

    return merged.drop(columns=["_merge"])


def build_analytical_dataset(
    ride_df: pd.DataFrame,
    product_df: pd.DataFrame,
    ride_address_df: pd.DataFrame,
    ride_estimative_path: Path,
    base_profile: dict,
    address_stats: dict,
) -> dict:
    log.info("=" * 60)
    log.info("Iniciando construcao do dataset analitico")
    log.info("=" * 60)

    prepare_output_dir(OUTPUT_DIR)

    validation = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(),
        "source_rows_rideestimative": base_profile["rows"],
        "source_chunks_rideestimative": base_profile["chunks"],
        "active_ride_ids": len(base_profile["active_ride_ids"]),
        "active_product_ids": len(base_profile["active_product_ids"]),
        "rows_written": 0,
        "chunks_written": 0,
        "missing_origin_rows": address_stats["missing_origin_rows"],
        "missing_destination_rows": address_stats["missing_destination_rows"],
        "output_dir": str(OUTPUT_DIR),
        "partition_row_counts": Counter(),
    }

    partitioning = None

    for chunk in pd.read_csv(
        ride_estimative_path,
        chunksize=CHUNKSIZE,
        **CSV_OPTIONS,
    ):
        chunk["ProductID"] = normalize_product_id(chunk["ProductID"])
        pre_join_rows = len(chunk)

        chunk = merge_dimension(chunk, ride_df, "RideID", "ride")
        chunk = merge_dimension(chunk, product_df, "ProductID", "product")
        chunk = merge_dimension(chunk, ride_address_df, "RideID", "rideaddress")

        if len(chunk) != pre_join_rows:
            raise ValueError(
                "O join alterou o volume do chunk. "
                f"Antes={pre_join_rows} | Depois={len(chunk)}"
            )

        chunk["CategoryID"] = pd.to_numeric(
            chunk["CategoryID"],
            errors="coerce",
        ).astype("Int64")

        if chunk["CategoryID"].isna().any():
            raise ValueError("Foram encontradas linhas com CategoryID nulo apos o join.")

        chunk = cast_object_columns_to_string(chunk)

        partition_counts = (
            chunk["CategoryID"]
            .astype("int64")
            .astype("string")
            .value_counts()
            .to_dict()
        )
        validation["partition_row_counts"].update(partition_counts)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if partitioning is None:
            partitioning = ds.partitioning(
                pa.schema([("CategoryID", table.schema.field("CategoryID").type)]),
                flavor="hive",
            )

        validation["chunks_written"] += 1
        ds.write_dataset(
            table,
            base_dir=OUTPUT_DIR,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"part-{validation['chunks_written']:05d}-{{i}}.parquet",
        )

        validation["rows_written"] += pre_join_rows
        log.info(
            "Chunk analitico #%s salvo | linhas=%s | acumulado=%s",
            validation["chunks_written"],
            pre_join_rows,
            validation["rows_written"],
        )

    if validation["rows_written"] != validation["source_rows_rideestimative"]:
        raise ValueError(
            "Volume final diferente do esperado. "
            f"Esperado={validation['source_rows_rideestimative']} | "
            f"Gravado={validation['rows_written']}"
        )

    parquet_dataset = ds.dataset(
        OUTPUT_DIR,
        format="parquet",
        partitioning="hive",
    )
    validation["rows_on_disk"] = parquet_dataset.count_rows()
    if validation["rows_on_disk"] != validation["rows_written"]:
        raise ValueError(
            "Volume no dataset Parquet nao confere com o volume processado. "
            f"Em disco={validation['rows_on_disk']} | "
            f"Processado={validation['rows_written']}"
        )

    validation["partition_row_counts"] = dict(
        sorted(validation["partition_row_counts"].items(), key=lambda item: int(item[0]))
    )
    validation["ready_for_eda"] = (
        validation["rows_written"] == validation["source_rows_rideestimative"]
        and validation["rows_on_disk"] == validation["rows_written"]
        and validation["missing_origin_rows"] == 0
        and validation["missing_destination_rows"] == 0
    )

    write_validation_report(validation)

    log.info("=" * 60)
    log.info("Dataset analitico consolidado com sucesso")
    log.info("Pronto para EDA/feature engineering: %s", validation["ready_for_eda"])
    log.info("=" * 60)
    return validation


# ============================================================
# MAIN
# ============================================================

def main() -> dict:
    log.info("=" * 60)
    log.info("Pipeline iniciado")
    log.info("=" * 60)

    base_profile = profile_rideestimative(RIDE_EST_FILE)
    ride_df = prepare_ride_dimension(base_profile["active_ride_ids"])
    product_df = prepare_product_dimension(base_profile["active_product_ids"])
    ride_address_df, address_stats = prepare_ride_address_dimension(
        base_profile["active_ride_ids"]
    )

    validation = build_analytical_dataset(
        ride_df=ride_df,
        product_df=product_df,
        ride_address_df=ride_address_df,
        ride_estimative_path=RIDE_EST_FILE,
        base_profile=base_profile,
        address_stats=address_stats,
    )

    log.info("=" * 60)
    log.info("Pipeline finalizado com sucesso")
    log.info("=" * 60)
    return validation


if __name__ == "__main__":
    main()
