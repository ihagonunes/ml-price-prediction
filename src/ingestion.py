import os
import logging
import pandas as pd
from datetime import datetime


DATA_PATH = "data"
RIDE_FILE = os.path.join(DATA_PATH, "ride.csv")
PRODUCT_FILE = os.path.join(DATA_PATH, "product.csv")
RIDE_ESTIMATE_FILE = os.path.join(DATA_PATH, "rideestimative.csv")

CHUNKSIZE = 100_000
TARGET_FIELD = "price"

PII_FIELDS = [
    "Name",
    "Phone",
    "Driver",
    "Plate",
    "DriverPhone",
    "DriverPicture",
    "Registration"
]
def setup_logging():
    os.makedirs("reports", exist_ok=True)
    log_filename = f"reports/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    logging.info("==============================================")
    logging.info("Pipeline iniciado")
    logging.info("==============================================")


def remove_pii(df):
    existing_pii = [col for col in PII_FIELDS if col in df.columns]

    if existing_pii:
        logging.warning(f"Removendo campos PII: {existing_pii}")
        df = df.drop(columns=existing_pii)

    return df


def validate_no_target_in_features(feature_columns):
    if TARGET_FIELD in feature_columns:
        raise ValueError(
            f"🚨 ERRO CRÍTICO: O campo '{TARGET_FIELD}' NÃO pode ser usado como feature!"
        )

    logging.info("Validação de target como feature OK.")


def read_csv_standard(filepath, name):
    logging.info(f"Lendo arquivo {name}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=";",
        on_bad_lines="warn"
    )

    logging.info(f"{name} carregado com sucesso.")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Colunas: {list(df.columns)}")

    df = remove_pii(df)

    return df


def read_csv_incremental(filepath, name):
    logging.info("==============================================")
    logging.info(f"Iniciando pipeline incremental {name}")
    logging.info("==============================================")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    total_rows = 0
    total_chunks = 0
    detected_schema = None

    for chunk in pd.read_csv(
    filepath,
    chunksize=CHUNKSIZE,
    sep=";",               
    low_memory=True
):

        total_chunks += 1

        if detected_schema is None:
            detected_schema = list(chunk.columns)
            logging.info(f"Schema detectado ({len(detected_schema)} colunas):")
            for col in detected_schema:
                logging.info(f"- {col}")

        logging.info(f"Processando chunk #{total_chunks}")

        chunk = remove_pii(chunk)

        feature_columns = [col for col in chunk.columns if col != TARGET_FIELD]
        validate_no_target_in_features(feature_columns)

        total_rows += len(chunk)

        logging.info(
            f"Chunk #{total_chunks} | Linhas: {len(chunk)} | Total acumulado: {total_rows}"
        )


    logging.info("==============================================")
    logging.info("Pipeline incremental concluído.")
    logging.info(f"Total linhas processadas: {total_rows}")
    logging.info(f"Total chunks: {total_chunks}")
    logging.info("==============================================")

    return detected_schema



def main():
    setup_logging()

    ride_df = read_csv_standard(RIDE_FILE, "ride.csv")
    product_df = read_csv_standard(PRODUCT_FILE, "product.csv")

    ride_estimate_schema = read_csv_incremental(
        RIDE_ESTIMATE_FILE,
        "rideestimative.csv"
    )

    logging.info("Pipeline finalizado com sucesso.")


if __name__ == "__main__":
    main()