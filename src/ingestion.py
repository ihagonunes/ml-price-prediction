
import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURAÇÃO
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

RIDE_FILE = DATA_DIR / "ride_v2.csv"
PRODUCT_FILE = DATA_DIR / "product.csv"
RIDE_EST_FILE = DATA_DIR / "rideestimative_v3.csv"
RIDE_ADDRESS_FILE = DATA_DIR / "rideaddress_v1.csv"

CHUNKSIZE = 100_000

PII_FIELDS = [
    "Name",
    "Phone",
    "Driver",
    "Plate",
    "DriverPhone",
    "DriverPicture",
    "Registration",
]

TARGET_FIELD = "price"

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
# UTILITÁRIOS
# ============================================================

def remove_pii_columns(columns):
    """Remove colunas PII ainda no momento da leitura."""
    return [col for col in columns if col not in PII_FIELDS]


def validate_no_target_feature(feature_columns):
    if TARGET_FIELD in feature_columns:
        raise ValueError(
            f"🚨 ERRO CRÍTICO: '{TARGET_FIELD}' não pode ser usado como feature."
        )
    log.info("Validação de uso de target como feature OK.")


# ============================================================
# LEITURA PADRÃO (ride + product)
# ============================================================

def read_standard_csv(filepath: Path, name: str):

    log.info(f"Lendo {name}...")

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    # Primeiro lê só cabeçalho para remover PII antes
    header = pd.read_csv(filepath, sep=";", nrows=0)
    cols = remove_pii_columns(header.columns)

    df = pd.read_csv(
        filepath,
        sep=";",
        usecols=cols,  # REMOVE PII AQUI
        low_memory=True,
    )

    log.info(f"{name} carregado | shape={df.shape}")
    return df


# ============================================================
# PIPELINE INCREMENTAL — rideestimative
# ============================================================

def read_rideestimative_incremental(filepath: Path):

    log.info("=" * 60)
    log.info("Iniciando pipeline incremental rideestimative_v3.csv")
    log.info("=" * 60)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    # Detecta schema real
    header = pd.read_csv(filepath, sep=";", nrows=0)
    real_columns = list(header.columns)

    log.info(f"Schema real detectado ({len(real_columns)} colunas):")
    for col in real_columns:
        log.info(f" - {col}")

    total_rows = 0
    total_chunks = 0

    for chunk in pd.read_csv(
        filepath,
        sep=";",
        chunksize=CHUNKSIZE,
        low_memory=True,
    ):
        total_chunks += 1
        rows = len(chunk)
        total_rows += rows

        log.info(
            f"Chunk #{total_chunks} | linhas={rows} | acumulado={total_rows}"
        )

    log.info("=" * 60)
    log.info("Pipeline incremental concluído")
    log.info(f"Total linhas processadas: {total_rows}")
    log.info(f"Total chunks: {total_chunks}")
    log.info("=" * 60)

    return {
        "rows": total_rows,
        "chunks": total_chunks,
        "columns": real_columns,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    log.info("=" * 60)
    log.info("Pipeline iniciado")
    log.info("=" * 60)

    ride_df = read_standard_csv(RIDE_FILE, "ride_v2.csv")
    product_df = read_standard_csv(PRODUCT_FILE, "product.csv")
    ride_address_df = read_standard_csv(
        RIDE_ADDRESS_FILE,
        "rideaddress_v1.csv"
    )

    # Bloqueio de uso de price como feature
    feature_cols = [c for c in ride_df.columns if c != TARGET_FIELD]
    validate_no_target_feature(feature_cols)

    stats = read_rideestimative_incremental(RIDE_EST_FILE)

    log.info("=" * 60)
    log.info("Pipeline finalizado com sucesso")
    log.info("=" * 60)

    return ride_df, product_df, ride_address_df, stats



if __name__ == "__main__":
    main()