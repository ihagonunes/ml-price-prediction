from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

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

from train import (
    CATEGORY_DATASETS,
    REPORTS_DIR,
    TemporalSplitConfig,
    compute_regression_metrics,
    get_feature_columns,
    load_category_feature_frame,
    load_temporal_frame,
)
from tune_selected_models import build_preprocessor

MODELS_DIR = PROJECT_ROOT / "models"
SERIALIZATION_REPORT_FILE = REPORTS_DIR / "model_serialization_report.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_tuned_params(category_id: int) -> tuple[dict, int]:
    params_file = REPORTS_DIR / "selected_model_tuning_best_params.csv"
    if not params_file.exists():
        raise FileNotFoundError(f"Best params file not found: {params_file}")

    params_df = pd.read_csv(params_file)
    row = params_df.loc[params_df["CategoryID"] == category_id].iloc[0]
    params = json.loads(row["BestParamsJSON"])
    num_boost_round = int(params.pop("num_boost_round"))
    return params, num_boost_round


def train_and_serialize(category_id: int, config: TemporalSplitConfig) -> dict:
    spec = CATEGORY_DATASETS[category_id]
    cat_name = spec["dataset_name"]
    log.info("Training final model for %s...", cat_name)

    category_frame = load_category_feature_frame(category_id, config)
    numeric_columns, categorical_columns = get_feature_columns(category_id, category_frame)

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    x_all = preprocessor.fit_transform(
        category_frame[numeric_columns + categorical_columns]
    )
    y_all = category_frame["Price"].astype("float64").to_numpy()

    params, num_boost_round = load_tuned_params(category_id)
    log.info("Training LightGBM with %d boosting rounds...", num_boost_round)

    dataset = lgb.Dataset(x_all, label=y_all, free_raw_data=False)
    model = lgb.train(
        params=params,
        train_set=dataset,
        num_boost_round=num_boost_round,
    )

    predictions = model.predict(x_all)
    metrics = compute_regression_metrics(pd.Series(y_all), predictions)

    model_bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "category_id": category_id,
        "category_name": cat_name,
        "num_boost_round": num_boost_round,
        "params": params,
    }

    output_file = MODELS_DIR / f"model_{cat_name.lower().replace(' ', '_')}.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_file, compress=3)

    loaded_bundle = joblib.load(output_file)
    sample_df = category_frame[numeric_columns + categorical_columns].head(5)
    x_sample = loaded_bundle["preprocessor"].transform(sample_df)
    test_predictions = loaded_bundle["model"].predict(x_sample)
    test_mae = float(np.mean(np.abs(y_all[:5] - test_predictions)))

    log.info("Model serialized: %s", output_file)
    log.info("Validation MAE (sample): %.4f", test_mae)

    return {
        "CategoryID": category_id,
        "CategoryName": cat_name,
        "ModelFile": str(output_file),
        "ModelType": "LightGBM (tuned)",
        "NumBoostRound": num_boost_round,
        "TrainingSamples": int(len(y_all)),
        "FeatureCount": int(x_all.shape[1]),
        "TrainingMAE": float(metrics["MAE"]),
        "TrainingRMSE": float(metrics["RMSE"]),
        "TrainingMAPE": float(metrics["MAPE"]),
        "TrainingR2": float(metrics["R2"]),
        "ValidationSampleMAE": test_mae,
        "SerializationOK": True,
    }


def write_report(results: list[dict]):
    df = pd.DataFrame(results)
    report_lines = [
        "# Model Serialization Report",
        "",
        f"- Generated: `{datetime.now().isoformat()}`",
        "- Framework: `joblib` with compression level 3",
        "- Model type: LightGBM (tuned via Optuna)",
        "",
        "## Serialized Models",
        "",
        df.to_markdown(index=False),
        "",
        "## How to Load and Use",
        "",
        "```python",
        "import joblib",
        "import pandas as pd",
        "",
        "bundle = joblib.load('models/model_uberx.joblib')",
        "preprocessor = bundle['preprocessor']",
        "model = bundle['model']",
        "",
        "x_input = preprocessor.transform(input_dataframe)",
        "predictions = model.predict(x_input)",
        "```",
        "",
        "## Input Requirements",
        "",
        "The input DataFrame must contain the same columns used during training.",
        "See the feature lists in `src/train.py` (NUMERIC_FEATURE_COLUMNS,",
        "BOOLEAN_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS) and the",
        "auxiliary cross-price columns defined in CATEGORY_DATASETS.",
        "",
        "## Validation",
        "",
        "Each model was loaded after serialization and tested with a sample",
        "of the training data to verify prediction capability.",
    ]

    SERIALIZATION_REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Report saved: %s", SERIALIZATION_REPORT_FILE)


def main():
    log.info("=" * 60)
    log.info("Serializing final tuned models")
    log.info("=" * 60)

    config = TemporalSplitConfig()
    _ = load_temporal_frame(config)

    results = []
    for category_id in sorted(CATEGORY_DATASETS):
        result = train_and_serialize(category_id, config)
        results.append(result)

    write_report(results)
    log.info("=" * 60)
    log.info("All 3 models serialized successfully")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
