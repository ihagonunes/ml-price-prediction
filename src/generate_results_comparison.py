from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from train import REPORTS_DIR


RESULTS_COMPARISON_FILE = REPORTS_DIR / "results_comparison.csv"
RESULTS_COMPARISON_REPORT_FILE = REPORTS_DIR / "results_comparison_report.md"

BASE_COLUMNS = [
    "CategoryID",
    "DatasetName",
    "Algorithm",
    "AlgorithmGroup",
    "ModelVariant",
    "RowType",
    "FoldName",
    "FoldsAveraged",
    "TrainRows",
    "EvaluationRows",
    "TrainStart",
    "TrainEnd",
    "EvaluationStart",
    "EvaluationEnd",
    "FeatureCount",
    "NumBoostRound",
    "TrialNumber",
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "MAEStd",
    "RMSEStd",
    "MAPEStd",
    "R2Std",
    "SourceFile",
    "GeneratedAt",
]

FOLD_SOURCE_FILES = [
    ("baseline", "default", REPORTS_DIR / "baseline_tscv_fold_metrics.csv"),
    ("advanced", "default", REPORTS_DIR / "uberx_advanced_tscv_fold_metrics.csv"),
    ("advanced", "default", REPORTS_DIR / "comfort_advanced_tscv_fold_metrics.csv"),
    ("advanced", "default", REPORTS_DIR / "black_advanced_tscv_fold_metrics.csv"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def read_required_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de metricas nao encontrado: {file_path}")
    return pd.read_csv(file_path)


def normalize_fold_metrics(
    frame: pd.DataFrame,
    algorithm_group: str,
    model_variant: str,
    source_file: Path,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["Algorithm"] = normalized["ModelName"]
    normalized["AlgorithmGroup"] = algorithm_group
    normalized["ModelVariant"] = model_variant
    normalized["RowType"] = "fold"
    normalized["FoldsAveraged"] = 1
    normalized["SourceFile"] = source_file.name
    normalized["GeneratedAt"] = datetime.now().isoformat()

    for column in ["NumBoostRound", "TrialNumber", "MAEStd", "RMSEStd", "MAPEStd", "R2Std"]:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    return normalized


def load_fold_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for algorithm_group, model_variant, source_file in FOLD_SOURCE_FILES:
        frame = read_required_csv(source_file)
        frames.append(
            normalize_fold_metrics(
                frame=frame,
                algorithm_group=algorithm_group,
                model_variant=model_variant,
                source_file=source_file,
            )
        )

    tuning_file = REPORTS_DIR / "selected_model_tuning_fold_metrics.csv"
    tuning_frame = read_required_csv(tuning_file)
    tuned_frame = tuning_frame.loc[tuning_frame["Phase"] == "tuned"].copy()
    tuned_frame["ModelName"] = "LightGBM_Tuned"
    frames.append(
        normalize_fold_metrics(
            frame=tuned_frame,
            algorithm_group="tuned",
            model_variant="optimized",
            source_file=tuning_file,
        )
    )

    fold_results = pd.concat(frames, ignore_index=True)
    return fold_results[BASE_COLUMNS].copy()


def build_mean_rows(fold_results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "CategoryID",
        "DatasetName",
        "Algorithm",
        "AlgorithmGroup",
        "ModelVariant",
        "FeatureCount",
        "NumBoostRound",
        "TrialNumber",
        "SourceFile",
    ]
    mean_rows = (
        fold_results.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            FoldsAveraged=("FoldName", "nunique"),
            TrainRows=("TrainRows", "mean"),
            EvaluationRows=("EvaluationRows", "mean"),
            TrainStart=("TrainStart", "min"),
            TrainEnd=("TrainEnd", "max"),
            EvaluationStart=("EvaluationStart", "min"),
            EvaluationEnd=("EvaluationEnd", "max"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean"),
            R2=("R2", "mean"),
            MAEStd=("MAE", "std"),
            RMSEStd=("RMSE", "std"),
            MAPEStd=("MAPE", "std"),
            R2Std=("R2", "std"),
        )
        .reset_index(drop=True)
    )
    mean_rows["RowType"] = "mean"
    mean_rows["FoldName"] = "mean"
    mean_rows["GeneratedAt"] = datetime.now().isoformat()
    return mean_rows[BASE_COLUMNS].copy()


def write_validation_report(results_df: pd.DataFrame) -> None:
    fold_rows = results_df.loc[results_df["RowType"] == "fold"].copy()
    mean_rows = results_df.loc[results_df["RowType"] == "mean"].copy()
    best_by_category = (
        mean_rows.sort_values(
            by=["CategoryID", "RMSE", "MAE"],
            kind="stable",
        )
        .groupby(["CategoryID", "DatasetName"], as_index=False)
        .first()
    )

    report_lines = [
        "# Results Comparison",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Arquivo principal: `{RESULTS_COMPARISON_FILE}`",
        f"- Linhas por fold: `{len(fold_rows)}`",
        f"- Linhas de media final: `{len(mean_rows)}`",
        f"- Categorias avaliadas: `{mean_rows['CategoryID'].nunique()}`",
        f"- Algoritmos por categoria: `{mean_rows.groupby('CategoryID')['Algorithm'].nunique().min()}` a `{mean_rows.groupby('CategoryID')['Algorithm'].nunique().max()}`",
        "",
        "## Melhores Modelos por RMSE Medio",
        "",
        best_by_category[
            [
                "CategoryID",
                "DatasetName",
                "Algorithm",
                "AlgorithmGroup",
                "ModelVariant",
                "RMSE",
                "MAE",
                "MAPE",
                "R2",
            ]
        ].to_markdown(index=False),
        "",
        "## Validacao",
        "",
        "- O arquivo consolida baselines, modelos avancados padrao e `LightGBM_Tuned`.",
        "- As linhas `fold` preservam as metricas originais de cada fold do TSCV.",
        "- As linhas `mean` calculam as medias finais e os desvios padrao das metricas por categoria e algoritmo.",
    ]
    RESULTS_COMPARISON_REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Gerando results_comparison.csv")
    log.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = load_fold_results()
    mean_results = build_mean_rows(fold_results)
    results_df = pd.concat([fold_results, mean_results], ignore_index=True)
    results_df = results_df.sort_values(
        by=["CategoryID", "AlgorithmGroup", "Algorithm", "RowType", "FoldName"],
        kind="stable",
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_COMPARISON_FILE, index=False)
    write_validation_report(results_df)

    log.info(
        "Arquivo gerado | linhas=%s | folds=%s | medias=%s",
        len(results_df),
        int((results_df["RowType"] == "fold").sum()),
        int((results_df["RowType"] == "mean").sum()),
    )
    log.info("Resultado salvo em %s", RESULTS_COMPARISON_FILE)
    log.info("=" * 60)
    return results_df


if __name__ == "__main__":
    main()
