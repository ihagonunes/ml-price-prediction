from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
MPLCONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_artifact import FinalLightGBMModel
from train import (
    BASE_DIR,
    CATEGORY_DATASETS,
    REPORTS_DIR,
    TARGET_COLUMN,
    TIME_ORDER_COLUMN,
    TemporalSplitConfig,
    get_feature_columns,
    load_category_feature_frame,
)


MODELS_DIR = BASE_DIR / "models"
TUNING_BEST_PARAMS_FILE = REPORTS_DIR / "selected_model_tuning_best_params.csv"
MODEL_VALIDATION_FILE = REPORTS_DIR / "final_model_serialization_validation.csv"
MODEL_REPORT_FILE = REPORTS_DIR / "final_model_serialization_report.md"

MODEL_FILE_NAMES = {
    2: "model_uberx.joblib",
    9: "model_comfort.joblib",
    4: "model_black.joblib",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def load_best_params() -> dict[int, dict[str, Any]]:
    if not TUNING_BEST_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Hiperparametros otimizados nao encontrados em {TUNING_BEST_PARAMS_FILE}."
        )

    params_df = pd.read_csv(TUNING_BEST_PARAMS_FILE)
    expected_categories = set(MODEL_FILE_NAMES)
    available_categories = set(params_df["CategoryID"].astype(int))
    missing_categories = sorted(expected_categories.difference(available_categories))
    if missing_categories:
        raise ValueError(
            f"Categorias sem hiperparametros otimizados: {missing_categories}"
        )

    best_params: dict[int, dict[str, Any]] = {}
    for row in params_df.itertuples(index=False):
        category_id = int(row.CategoryID)
        if category_id not in expected_categories:
            continue
        params = json.loads(row.BestParamsJSON)
        num_boost_round = int(params.pop("num_boost_round"))
        best_params[category_id] = {
            "best_trial_number": int(row.BestTrialNumber),
            "trials_executed": int(row.TrialsExecuted),
            "num_boost_round": num_boost_round,
            "params": params,
            "tuned_mean_rmse": float(row.TunedMeanRMSE),
            "tuned_mean_mae": float(row.TunedMeanMAE),
            "tuned_mean_mape": float(row.TunedMeanMAPE),
            "tuned_mean_r2": float(row.TunedMeanR2),
        }
    return best_params


def train_final_model(
    category_id: int,
    config: TemporalSplitConfig,
    best_params: dict[str, Any],
) -> tuple[FinalLightGBMModel, pd.DataFrame, dict[str, Any]]:
    category_frame = load_category_feature_frame(category_id, config)
    numeric_columns, categorical_columns = get_feature_columns(category_id, category_frame)
    feature_columns = numeric_columns + categorical_columns

    x_train = category_frame[feature_columns].copy()
    y_train = category_frame[TARGET_COLUMN].astype("float64").to_numpy()
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    x_train_prepared = preprocessor.fit_transform(x_train)

    train_dataset = lgb.Dataset(x_train_prepared, label=y_train, free_raw_data=False)
    booster = lgb.train(
        params=best_params["params"],
        train_set=train_dataset,
        num_boost_round=best_params["num_boost_round"],
    )

    metadata = {
        "category_id": category_id,
        "category_name": CATEGORY_DATASETS[category_id]["dataset_name"],
        "model_name": "LightGBM",
        "training_rows": int(len(category_frame)),
        "feature_count": int(len(preprocessor.get_feature_names_out())),
        "source_file": CATEGORY_DATASETS[category_id]["file_name"],
        "training_start_date": category_frame[TIME_ORDER_COLUMN].min().date().isoformat(),
        "training_end_date": category_frame[TIME_ORDER_COLUMN].max().date().isoformat(),
        "target_column": TARGET_COLUMN,
        "time_order_column": TIME_ORDER_COLUMN,
        "best_trial_number": best_params["best_trial_number"],
        "trials_executed": best_params["trials_executed"],
        "num_boost_round": best_params["num_boost_round"],
        "tuned_mean_rmse": best_params["tuned_mean_rmse"],
        "tuned_mean_mae": best_params["tuned_mean_mae"],
        "tuned_mean_mape": best_params["tuned_mean_mape"],
        "tuned_mean_r2": best_params["tuned_mean_r2"],
        "serialized_at": datetime.now().isoformat(),
    }
    artifact = FinalLightGBMModel(
        category_id=category_id,
        category_name=CATEGORY_DATASETS[category_id]["dataset_name"],
        model_name="LightGBM",
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocessor=preprocessor,
        booster=booster,
        metadata=metadata,
    )
    return artifact, category_frame, metadata


def validate_serialized_model(
    model_path: Path,
    category_frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    loaded_model: FinalLightGBMModel = joblib.load(model_path)
    example_frame = category_frame[loaded_model.feature_columns].head(1).copy()
    prediction = loaded_model.predict(example_frame)[0]
    prediction_is_finite = bool(np.isfinite(prediction))
    if not prediction_is_finite:
        raise ValueError(f"Predicao invalida gerada pelo modelo {model_path}")

    return {
        "CategoryID": loaded_model.category_id,
        "DatasetName": loaded_model.category_name,
        "ModelName": loaded_model.model_name,
        "ModelPath": str(model_path),
        "ModelFileName": model_path.name,
        "ModelFileSizeMB": round(model_path.stat().st_size / (1024 * 1024), 4),
        "TrainingRows": metadata["training_rows"],
        "FeatureCount": metadata["feature_count"],
        "TrainingStartDate": metadata["training_start_date"],
        "TrainingEndDate": metadata["training_end_date"],
        "ExampleRows": int(len(example_frame)),
        "ExampleTargetPrice": float(category_frame[TARGET_COLUMN].head(1).iloc[0]),
        "ExamplePrediction": float(prediction),
        "PredictionIsFinite": prediction_is_finite,
        "LoadedArtifactClass": loaded_model.__class__.__name__,
        "ValidatedAt": datetime.now().isoformat(),
    }


def write_report(validation_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(MODEL_VALIDATION_FILE, index=False)

    report_lines = [
        "# Serializacao dos Modelos Finais",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Diretorio dos modelos: `{MODELS_DIR}`",
        "- Formato: `.joblib` com artefato contendo preprocessor, booster LightGBM, lista de features e metadados.",
        "",
        "## Validacao de Reload e Predicao",
        "",
        validation_df.to_markdown(index=False),
        "",
        "## Resultado",
        "",
        "- Os tres modelos finais otimizados foram serializados em `/models/`.",
        "- Cada arquivo foi recarregado com `joblib.load` e gerou uma predicao finita a partir de um exemplo real de input.",
        "- Os arquivos `.joblib` permanecem ignorados pelo Git; o script versionavel recria os artefatos quando necessario.",
    ]
    MODEL_REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Serializando modelos finais otimizados")
    log.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config = TemporalSplitConfig()
    best_params_by_category = load_best_params()

    validation_rows: list[dict[str, Any]] = []
    for category_id in [2, 9, 4]:
        model_path = MODELS_DIR / MODEL_FILE_NAMES[category_id]
        artifact, category_frame, metadata = train_final_model(
            category_id=category_id,
            config=config,
            best_params=best_params_by_category[category_id],
        )
        joblib.dump(artifact, model_path, compress=3)
        validation_row = validate_serialized_model(model_path, category_frame, metadata)
        validation_rows.append(validation_row)
        log.info(
            "Modelo serializado e validado | categoria=%s | arquivo=%s | predicao=%.4f",
            artifact.category_name,
            model_path.name,
            validation_row["ExamplePrediction"],
        )

    validation_df = pd.DataFrame(validation_rows)
    write_report(validation_df)

    log.info("Validacao salva em %s", MODEL_VALIDATION_FILE)
    log.info("=" * 60)
    return validation_df


if __name__ == "__main__":
    main()
