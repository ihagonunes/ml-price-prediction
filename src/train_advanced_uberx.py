from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MPLCONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from train import (
    BASELINE_SUMMARY_FILE,
    CATEGORY_DATASETS,
    REPORTS_DIR,
    TemporalSplitConfig,
    apply_split_window_to_category_frame,
    build_split_windows,
    compute_regression_metrics,
    get_feature_columns,
    load_category_feature_frame,
    load_temporal_frame,
)


ADVANCED_FOLD_METRICS_FILE = REPORTS_DIR / "uberx_advanced_tscv_fold_metrics.csv"
ADVANCED_SUMMARY_FILE = REPORTS_DIR / "uberx_advanced_tscv_summary.csv"
ADVANCED_COMPARISON_FILE = REPORTS_DIR / "uberx_advanced_vs_baseline.csv"
ADVANCED_REPORT_FILE = REPORTS_DIR / "uberx_advanced_tscv_report.md"
UBERX_CATEGORY_ID = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def build_advanced_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[ColumnTransformer, dict[str, object]]:
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    model_registry = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        ),
        "LightGBM": "native_lightgbm",
        "RandomForest": RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),
    }
    return preprocessor, model_registry


def load_uberx_baseline_summary() -> tuple[pd.DataFrame, pd.Series]:
    if not BASELINE_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Resumo baseline nao encontrado em {BASELINE_SUMMARY_FILE}. Execute src/train.py antes."
        )

    summary_df = pd.read_csv(BASELINE_SUMMARY_FILE)
    uberx_df = summary_df.loc[summary_df["CategoryID"] == UBERX_CATEGORY_ID].copy()
    if uberx_df.empty:
        raise ValueError("Nao ha linhas de baseline para UberX no resumo atual.")

    best_row = uberx_df.sort_values(
        by=["MeanMAE", "MeanRMSE"],
        kind="stable",
    ).iloc[0]
    return uberx_df.reset_index(drop=True), best_row


def run_advanced_tscv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = TemporalSplitConfig()
    temporal_frame = load_temporal_frame(config)
    cv_windows, _ = build_split_windows(temporal_frame, config)
    dataset_spec = CATEGORY_DATASETS[UBERX_CATEGORY_ID]
    category_frame = load_category_feature_frame(UBERX_CATEGORY_ID, config)
    numeric_columns, categorical_columns = get_feature_columns(
        UBERX_CATEGORY_ID,
        category_frame,
    )

    fold_rows: list[dict] = []
    for split_window in cv_windows:
        train_frame, evaluation_frame = apply_split_window_to_category_frame(
            category_frame,
            split_window,
        )

        x_train = train_frame[numeric_columns + categorical_columns].copy()
        y_train = train_frame["Price"].astype("float64")
        x_evaluation = evaluation_frame[numeric_columns + categorical_columns].copy()
        y_evaluation = evaluation_frame["Price"].astype("float64")

        preprocessor, model_registry = build_advanced_models(
            numeric_columns,
            categorical_columns,
        )
        x_train_prepared = preprocessor.fit_transform(x_train)
        x_evaluation_prepared = preprocessor.transform(x_evaluation)
        feature_count = len(preprocessor.get_feature_names_out())

        for model_name, model_object in model_registry.items():
            if model_name == "LightGBM":
                train_dataset = lgb.Dataset(x_train_prepared, label=y_train.to_numpy())
                model_object = lgb.train(
                    params={
                        "objective": "regression",
                        "metric": "l2",
                        "verbosity": -1,
                        "seed": 42,
                        "num_threads": -1,
                    },
                    train_set=train_dataset,
                    num_boost_round=100,
                )
                predictions = model_object.predict(x_evaluation_prepared)
            else:
                model_object.fit(x_train_prepared, y_train)
                predictions = model_object.predict(x_evaluation_prepared)

            metrics = compute_regression_metrics(y_evaluation, predictions)

            fold_rows.append(
                {
                    "CategoryID": UBERX_CATEGORY_ID,
                    "DatasetName": dataset_spec["dataset_name"],
                    "ModelName": model_name,
                    "FoldName": split_window.name,
                    "TrainRows": int(len(train_frame)),
                    "EvaluationRows": int(len(evaluation_frame)),
                    "TrainStart": split_window.train_start.date().isoformat(),
                    "TrainEnd": split_window.train_end.date().isoformat(),
                    "EvaluationStart": split_window.evaluation_start.date().isoformat(),
                    "EvaluationEnd": split_window.evaluation_end.date().isoformat(),
                    "FeatureCount": int(feature_count),
                    **metrics,
                }
            )
            log.info(
                "Avancado treinado | modelo=%s | fold=%s | MAE=%.4f | RMSE=%.4f | MAPE=%.2f | R2=%.4f",
                model_name,
                split_window.name,
                metrics["MAE"],
                metrics["RMSE"],
                metrics["MAPE"],
                metrics["R2"],
            )

    fold_metrics_df = pd.DataFrame(fold_rows).sort_values(
        by=["ModelName", "FoldName"],
        kind="stable",
    ).reset_index(drop=True)
    summary_df = (
        fold_metrics_df.groupby(["CategoryID", "DatasetName", "ModelName"], as_index=False)
        .agg(
            Folds=("FoldName", "count"),
            MeanMAE=("MAE", "mean"),
            StdMAE=("MAE", "std"),
            MeanRMSE=("RMSE", "mean"),
            StdRMSE=("RMSE", "std"),
            MeanMAPE=("MAPE", "mean"),
            StdMAPE=("MAPE", "std"),
            MeanR2=("R2", "mean"),
            StdR2=("R2", "std"),
            MeanFeatureCount=("FeatureCount", "mean"),
        )
        .sort_values(by=["MeanMAE", "MeanRMSE"], kind="stable")
        .reset_index(drop=True)
    )

    baseline_df, best_baseline_row = load_uberx_baseline_summary()
    comparison_df = summary_df.copy()
    comparison_df["BestBaselineModel"] = best_baseline_row["ModelName"]
    comparison_df["BestBaselineMAE"] = float(best_baseline_row["MeanMAE"])
    comparison_df["BestBaselineRMSE"] = float(best_baseline_row["MeanRMSE"])
    comparison_df["BestBaselineMAPE"] = float(best_baseline_row["MeanMAPE"])
    comparison_df["BestBaselineR2"] = float(best_baseline_row["MeanR2"])
    comparison_df["DeltaMAEvsBestBaseline"] = (
        comparison_df["MeanMAE"] - comparison_df["BestBaselineMAE"]
    )
    comparison_df["DeltaRMSEvsBestBaseline"] = (
        comparison_df["MeanRMSE"] - comparison_df["BestBaselineRMSE"]
    )
    comparison_df["DeltaMAPEvsBestBaseline"] = (
        comparison_df["MeanMAPE"] - comparison_df["BestBaselineMAPE"]
    )
    comparison_df["DeltaR2vsBestBaseline"] = (
        comparison_df["MeanR2"] - comparison_df["BestBaselineR2"]
    )
    comparison_df["BeatsBestBaselineMAE"] = (
        comparison_df["MeanMAE"] < comparison_df["BestBaselineMAE"]
    )

    return fold_metrics_df, summary_df, comparison_df


def write_report(
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(ADVANCED_FOLD_METRICS_FILE, index=False)
    summary_df.to_csv(ADVANCED_SUMMARY_FILE, index=False)
    comparison_df.to_csv(ADVANCED_COMPARISON_FILE, index=False)

    best_row = comparison_df.sort_values(by=["MeanMAE", "MeanRMSE"], kind="stable").iloc[0]
    report_lines = [
        "# Modelos Avancados para UberX com TSCV",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Dataset: `{CATEGORY_DATASETS[UBERX_CATEGORY_ID]['file_name']}`",
        "- Modelos treinados: `XGBoost`, `LightGBM` e `RandomForest` com hiperparametros padrao.",
        "",
        "## Resultado DE",
        "",
        "- O mesmo TSCV da etapa baseline foi reutilizado para UberX, preservando comparabilidade temporal.",
        "- As metricas por fold e as medias gerais foram persistidas em CSV, junto com o comparativo contra o melhor baseline anterior.",
        "",
        "## Resultado DS",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Comparacao com o Melhor Baseline de UberX",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Leitura DS",
        "",
        (
            f"- O melhor modelo avancado por MAE foi `{best_row['ModelName']}` "
            f"com MAE medio `{best_row['MeanMAE']:.4f}`, RMSE `{best_row['MeanRMSE']:.4f}`, "
            f"MAPE `{best_row['MeanMAPE']:.2f}%` e R2 `{best_row['MeanR2']:.4f}`."
        ),
        (
            f"- Comparado ao melhor baseline (`{best_row['BestBaselineModel']}`), "
            f"o delta de MAE foi `{best_row['DeltaMAEvsBestBaseline']:.4f}` e o delta de R2 foi "
            f"`{best_row['DeltaR2vsBestBaseline']:.4f}`."
        ),
        "",
        "## Detalhe por Fold",
        "",
        fold_metrics_df.to_markdown(index=False),
    ]
    ADVANCED_REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Relatorios avancados salvos em %s", REPORTS_DIR)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("=" * 60)
    log.info("Treinando modelos avancados para UberX")
    log.info("=" * 60)

    fold_metrics_df, summary_df, comparison_df = run_advanced_tscv()
    write_report(
        fold_metrics_df=fold_metrics_df,
        summary_df=summary_df,
        comparison_df=comparison_df,
    )

    log.info("=" * 60)
    log.info("Modelos avancados de UberX concluidos")
    log.info("=" * 60)
    return fold_metrics_df, summary_df, comparison_df


if __name__ == "__main__":
    main()
