from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
MPLCONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train import (
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


SELECTED_MODEL_FILE = REPORTS_DIR / "best_model_by_category.csv"
TUNING_TRIALS_FILE = REPORTS_DIR / "selected_model_tuning_trial_metrics.csv"
TUNING_FOLD_METRICS_FILE = REPORTS_DIR / "selected_model_tuning_fold_metrics.csv"
TUNING_SUMMARY_FILE = REPORTS_DIR / "selected_model_tuning_summary.csv"
TUNING_BEST_PARAMS_FILE = REPORTS_DIR / "selected_model_tuning_best_params.csv"
TUNING_REPORT_FILE = REPORTS_DIR / "selected_model_tuning_report.md"

TUNING_TRIALS_PER_CATEGORY = 15
DEFAULT_NUM_BOOST_ROUND = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldPreparedData:
    category_id: int
    dataset_name: str
    fold_name: str
    train_rows: int
    evaluation_rows: int
    train_start: str
    train_end: str
    evaluation_start: str
    evaluation_end: str
    feature_count: int
    x_train: Any
    y_train: Any
    x_evaluation: Any
    y_evaluation: Any


def get_selected_categories() -> list[int]:
    if not SELECTED_MODEL_FILE.exists():
        return sorted(CATEGORY_DATASETS)

    selection_df = pd.read_csv(SELECTED_MODEL_FILE)
    selected_categories = (
        selection_df.loc[selection_df["SelectedModel"] == "LightGBM", "CategoryID"]
        .dropna()
        .astype(int)
        .tolist()
    )
    if not selected_categories:
        raise ValueError(
            "Nenhuma categoria com LightGBM selecionado encontrada em best_model_by_category.csv."
        )
    return sorted(selected_categories)


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


def build_fold_cache(
    category_id: int,
    config: TemporalSplitConfig,
    cv_windows: list,
) -> list[FoldPreparedData]:
    category_frame = load_category_feature_frame(category_id, config)
    numeric_columns, categorical_columns = get_feature_columns(category_id, category_frame)
    dataset_name = CATEGORY_DATASETS[category_id]["dataset_name"]

    prepared_folds: list[FoldPreparedData] = []
    for split_window in cv_windows:
        train_frame, evaluation_frame = apply_split_window_to_category_frame(
            category_frame,
            split_window,
        )

        x_train = train_frame[numeric_columns + categorical_columns].copy()
        y_train = train_frame["Price"].astype("float64").to_numpy()
        x_evaluation = evaluation_frame[numeric_columns + categorical_columns].copy()
        y_evaluation = evaluation_frame["Price"].astype("float64").to_numpy()

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        x_train_prepared = preprocessor.fit_transform(x_train)
        x_evaluation_prepared = preprocessor.transform(x_evaluation)
        feature_count = len(preprocessor.get_feature_names_out())

        prepared_folds.append(
            FoldPreparedData(
                category_id=category_id,
                dataset_name=dataset_name,
                fold_name=split_window.name,
                train_rows=int(len(train_frame)),
                evaluation_rows=int(len(evaluation_frame)),
                train_start=split_window.train_start.date().isoformat(),
                train_end=split_window.train_end.date().isoformat(),
                evaluation_start=split_window.evaluation_start.date().isoformat(),
                evaluation_end=split_window.evaluation_end.date().isoformat(),
                feature_count=int(feature_count),
                x_train=x_train_prepared,
                y_train=y_train,
                x_evaluation=x_evaluation_prepared,
                y_evaluation=y_evaluation,
            )
        )

    return prepared_folds


def get_default_lightgbm_params() -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "seed": 42,
        "num_threads": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def sample_lightgbm_params(trial: optuna.Trial) -> tuple[dict[str, Any], int]:
    params = get_default_lightgbm_params()
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200, step=10),
            "feature_fraction": trial.suggest_float(
                "feature_fraction",
                0.6,
                1.0,
                step=0.05,
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction",
                0.6,
                1.0,
                step=0.05,
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
    )
    num_boost_round = trial.suggest_int("num_boost_round", 100, 400, step=50)
    return params, num_boost_round


def evaluate_lightgbm_configuration(
    category_id: int,
    folds: list[FoldPreparedData],
    params: dict[str, Any],
    num_boost_round: int,
    phase: str,
    trial_number: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_dataset = lgb.Dataset(fold.x_train, label=fold.y_train, free_raw_data=False)
        model = lgb.train(
            params=params,
            train_set=train_dataset,
            num_boost_round=num_boost_round,
        )
        predictions = model.predict(fold.x_evaluation)
        metrics = compute_regression_metrics(fold.y_evaluation, predictions)
        fold_rows.append(
            {
                "CategoryID": category_id,
                "DatasetName": fold.dataset_name,
                "ModelName": "LightGBM",
                "Phase": phase,
                "TrialNumber": trial_number,
                "FoldName": fold.fold_name,
                "TrainRows": fold.train_rows,
                "EvaluationRows": fold.evaluation_rows,
                "TrainStart": fold.train_start,
                "TrainEnd": fold.train_end,
                "EvaluationStart": fold.evaluation_start,
                "EvaluationEnd": fold.evaluation_end,
                "FeatureCount": fold.feature_count,
                "NumBoostRound": int(num_boost_round),
                **metrics,
            }
        )

    fold_metrics_df = pd.DataFrame(fold_rows)
    summary = {
        "CategoryID": category_id,
        "DatasetName": CATEGORY_DATASETS[category_id]["dataset_name"],
        "ModelName": "LightGBM",
        "Phase": phase,
        "TrialNumber": trial_number,
        "Folds": int(len(fold_rows)),
        "MeanMAE": float(fold_metrics_df["MAE"].mean()),
        "StdMAE": float(fold_metrics_df["MAE"].std(ddof=1)),
        "MeanRMSE": float(fold_metrics_df["RMSE"].mean()),
        "StdRMSE": float(fold_metrics_df["RMSE"].std(ddof=1)),
        "MeanMAPE": float(fold_metrics_df["MAPE"].mean()),
        "StdMAPE": float(fold_metrics_df["MAPE"].std(ddof=1)),
        "MeanR2": float(fold_metrics_df["R2"].mean()),
        "StdR2": float(fold_metrics_df["R2"].std(ddof=1)),
        "MeanFeatureCount": float(fold_metrics_df["FeatureCount"].mean()),
        "NumBoostRound": int(num_boost_round),
        "ParamsJSON": json.dumps(
            {
                **params,
                "num_boost_round": int(num_boost_round),
            },
            sort_keys=True,
        ),
    }
    return fold_rows, summary


def run_category_tuning(
    category_id: int,
    folds: list[FoldPreparedData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    default_params = get_default_lightgbm_params()
    default_fold_rows, default_summary = evaluate_lightgbm_configuration(
        category_id=category_id,
        folds=folds,
        params=default_params,
        num_boost_round=DEFAULT_NUM_BOOST_ROUND,
        phase="default",
    )
    log.info(
        "Baseline LightGBM | categoria=%s | RMSE=%.4f | MAE=%.4f",
        CATEGORY_DATASETS[category_id]["dataset_name"],
        default_summary["MeanRMSE"],
        default_summary["MeanMAE"],
    )

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = sample_lightgbm_params(trial)
        _, summary = evaluate_lightgbm_configuration(
            category_id=category_id,
            folds=folds,
            params=params,
            num_boost_round=num_boost_round,
            phase="tuning",
            trial_number=trial.number,
        )
        trial.set_user_attr("MeanMAE", summary["MeanMAE"])
        trial.set_user_attr("MeanMAPE", summary["MeanMAPE"])
        trial.set_user_attr("MeanR2", summary["MeanR2"])
        trial.set_user_attr("StdRMSE", summary["StdRMSE"])
        trial.set_user_attr("StdMAE", summary["StdMAE"])
        trial.set_user_attr("NumBoostRound", summary["NumBoostRound"])
        trial.set_user_attr("ParamsJSON", summary["ParamsJSON"])
        return summary["MeanRMSE"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=TUNING_TRIALS_PER_CATEGORY, show_progress_bar=False)

    for trial in study.trials:
        trial_rows.append(
            {
                "CategoryID": category_id,
                "DatasetName": CATEGORY_DATASETS[category_id]["dataset_name"],
                "ModelName": "LightGBM",
                "TrialNumber": int(trial.number),
                "TrialState": trial.state.name,
                "MeanRMSE": float(trial.value) if trial.value is not None else None,
                "MeanMAE": trial.user_attrs.get("MeanMAE"),
                "MeanMAPE": trial.user_attrs.get("MeanMAPE"),
                "MeanR2": trial.user_attrs.get("MeanR2"),
                "StdRMSE": trial.user_attrs.get("StdRMSE"),
                "StdMAE": trial.user_attrs.get("StdMAE"),
                "NumBoostRound": trial.user_attrs.get("NumBoostRound"),
                "ParamsJSON": trial.user_attrs.get("ParamsJSON"),
                "IsBestTrial": trial.number == study.best_trial.number,
            }
        )

    best_params = get_default_lightgbm_params()
    best_params.update(
        {
            key: value
            for key, value in study.best_trial.params.items()
            if key != "num_boost_round"
        }
    )
    best_num_boost_round = int(study.best_trial.params["num_boost_round"])
    tuned_fold_rows, tuned_summary = evaluate_lightgbm_configuration(
        category_id=category_id,
        folds=folds,
        params=best_params,
        num_boost_round=best_num_boost_round,
        phase="tuned",
        trial_number=study.best_trial.number,
    )

    best_params_row = {
        "CategoryID": category_id,
        "DatasetName": CATEGORY_DATASETS[category_id]["dataset_name"],
        "ModelName": "LightGBM",
        "BestTrialNumber": int(study.best_trial.number),
        "TrialsExecuted": int(len(study.trials)),
        "BestNumBoostRound": int(best_num_boost_round),
        "BestParamsJSON": json.dumps(
            {
                **best_params,
                "num_boost_round": best_num_boost_round,
            },
            sort_keys=True,
        ),
        "DefaultMeanRMSE": float(default_summary["MeanRMSE"]),
        "TunedMeanRMSE": float(tuned_summary["MeanRMSE"]),
        "DeltaRMSE": float(tuned_summary["MeanRMSE"] - default_summary["MeanRMSE"]),
        "DefaultMeanMAE": float(default_summary["MeanMAE"]),
        "TunedMeanMAE": float(tuned_summary["MeanMAE"]),
        "DeltaMAE": float(tuned_summary["MeanMAE"] - default_summary["MeanMAE"]),
        "DefaultMeanMAPE": float(default_summary["MeanMAPE"]),
        "TunedMeanMAPE": float(tuned_summary["MeanMAPE"]),
        "DeltaMAPE": float(tuned_summary["MeanMAPE"] - default_summary["MeanMAPE"]),
        "DefaultMeanR2": float(default_summary["MeanR2"]),
        "TunedMeanR2": float(tuned_summary["MeanR2"]),
        "DeltaR2": float(tuned_summary["MeanR2"] - default_summary["MeanR2"]),
    }

    log.info(
        "Tuning concluido | categoria=%s | melhor_trial=%s | RMSE default=%.4f | RMSE tuned=%.4f",
        CATEGORY_DATASETS[category_id]["dataset_name"],
        study.best_trial.number,
        default_summary["MeanRMSE"],
        tuned_summary["MeanRMSE"],
    )

    return (
        default_fold_rows + tuned_fold_rows,
        trial_rows,
        best_params_row,
        tuned_summary,
    )


def write_report(
    fold_metrics_df: pd.DataFrame,
    trial_metrics_df: pd.DataFrame,
    best_params_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(TUNING_FOLD_METRICS_FILE, index=False)
    trial_metrics_df.to_csv(TUNING_TRIALS_FILE, index=False)
    best_params_df.to_csv(TUNING_BEST_PARAMS_FILE, index=False)

    summary_df = best_params_df[
        [
            "CategoryID",
            "DatasetName",
            "BestTrialNumber",
            "TrialsExecuted",
            "DefaultMeanRMSE",
            "TunedMeanRMSE",
            "DeltaRMSE",
            "DefaultMeanMAE",
            "TunedMeanMAE",
            "DeltaMAE",
            "DefaultMeanMAPE",
            "TunedMeanMAPE",
            "DeltaMAPE",
            "DefaultMeanR2",
            "TunedMeanR2",
            "DeltaR2",
        ]
    ].copy()
    summary_df.to_csv(TUNING_SUMMARY_FILE, index=False)

    report_lines = [
        "# Otimizacao de Hiperparametros dos Modelos Selecionados",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Metodo: `Optuna TPE` com `{TUNING_TRIALS_PER_CATEGORY}` trials por categoria.",
        "- Modelo otimizado: `LightGBM` nas categorias `UberX`, `Uber Comfort` e `Uber Black`.",
        "- Criterio objetivo: `menor RMSE medio no TSCV`.",
        "",
        "## Resumo de Ganho",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Melhores Hiperparametros",
        "",
    ]

    for row in best_params_df.itertuples(index=False):
        report_lines.extend(
            [
                f"### {row.DatasetName}",
                "",
                f"- Melhor trial: `{row.BestTrialNumber}` de `{row.TrialsExecuted}`.",
                f"- RMSE default vs tuned: `{row.DefaultMeanRMSE:.4f}` -> `{row.TunedMeanRMSE:.4f}` (delta `{row.DeltaRMSE:.4f}`).",
                f"- MAE default vs tuned: `{row.DefaultMeanMAE:.4f}` -> `{row.TunedMeanMAE:.4f}` (delta `{row.DeltaMAE:.4f}`).",
                f"- MAPE default vs tuned: `{row.DefaultMeanMAPE:.2f}%` -> `{row.TunedMeanMAPE:.2f}%` (delta `{row.DeltaMAPE:.2f}`).",
                f"- R2 default vs tuned: `{row.DefaultMeanR2:.4f}` -> `{row.TunedMeanR2:.4f}` (delta `{row.DeltaR2:.4f}`).",
                f"- Hiperparametros: `{row.BestParamsJSON}`.",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Leitura DS",
            "",
            "- O ganho principal foi avaliado por RMSE medio, que continua sendo a metrica de selecao adotada no projeto.",
            "- Melhorias simultaneas em RMSE e MAE indicam calibracao mais robusta do erro absoluto; quando o ganho em MAPE for menor, isso sugere que a distribuicao relativa do erro ja estava bem capturada no modelo padrao.",
            "- Se alguma categoria mostrar ganho marginal, isso sinaliza que o modelo default ja estava perto de um bom regime e que o proximo salto pode depender mais de novas features do que de tuning adicional.",
        ]
    )

    TUNING_REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    log.info("=" * 60)
    log.info("Otimizando hiperparametros dos modelos selecionados")
    log.info("=" * 60)

    config = TemporalSplitConfig()
    temporal_frame = load_temporal_frame(config)
    cv_windows, _ = build_split_windows(temporal_frame, config)
    selected_categories = get_selected_categories()

    all_fold_rows: list[dict[str, Any]] = []
    all_trial_rows: list[dict[str, Any]] = []
    best_params_rows: list[dict[str, Any]] = []

    for category_id in selected_categories:
        log.info(
            "Preparando folds para tuning | categoria=%s",
            CATEGORY_DATASETS[category_id]["dataset_name"],
        )
        folds = build_fold_cache(category_id, config, cv_windows)
        fold_rows, trial_rows, best_params_row, _ = run_category_tuning(category_id, folds)
        all_fold_rows.extend(fold_rows)
        all_trial_rows.extend(trial_rows)
        best_params_rows.append(best_params_row)

    fold_metrics_df = pd.DataFrame(all_fold_rows)
    trial_metrics_df = pd.DataFrame(all_trial_rows).sort_values(
        by=["CategoryID", "MeanRMSE", "TrialNumber"],
        kind="stable",
    ).reset_index(drop=True)
    best_params_df = pd.DataFrame(best_params_rows).sort_values(
        by="CategoryID",
        kind="stable",
    ).reset_index(drop=True)

    write_report(fold_metrics_df, trial_metrics_df, best_params_df)

    log.info(
        "Tuning concluido | categorias=%s | trials=%s",
        len(selected_categories),
        len(trial_metrics_df),
    )
    log.info("Relatorios salvos em %s", REPORTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
