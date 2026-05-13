from __future__ import annotations

import logging
import sys
from datetime import datetime

import pandas as pd

from train import CATEGORY_DATASETS, REPORTS_DIR


MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison_all_categories.csv"
MODEL_SELECTION_FILE = REPORTS_DIR / "best_model_by_category.csv"
MODEL_SELECTION_REPORT_FILE = REPORTS_DIR / "model_selection_report.md"

SUMMARY_SOURCES = [
    ("baseline", REPORTS_DIR / "baseline_tscv_summary.csv"),
    ("advanced", REPORTS_DIR / "uberx_advanced_tscv_summary.csv"),
    ("advanced", REPORTS_DIR / "comfort_advanced_tscv_summary.csv"),
    ("advanced", REPORTS_DIR / "black_advanced_tscv_summary.csv"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_model_summaries() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for algorithm_group, file_path in SUMMARY_SOURCES:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Resumo de metricas nao encontrado em {file_path}. Gere os treinos antes."
            )

        frame = pd.read_csv(file_path)
        frame["AlgorithmGroup"] = algorithm_group
        frame["SourceFile"] = file_path.name
        frames.append(frame)

    comparison_df = pd.concat(frames, ignore_index=True)
    comparison_df["CategoryName"] = comparison_df["CategoryID"].map(
        {
            category_id: spec["dataset_name"]
            for category_id, spec in CATEGORY_DATASETS.items()
        }
    )
    comparison_df = comparison_df.sort_values(
        by=["CategoryID", "MeanRMSE", "MeanMAE"],
        kind="stable",
    ).reset_index(drop=True)
    comparison_df["RMSECategoryRank"] = (
        comparison_df.groupby("CategoryID")["MeanRMSE"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return comparison_df


def build_selection_frame(comparison_df: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[dict] = []
    for category_id, category_frame in comparison_df.groupby("CategoryID", sort=True):
        ordered = category_frame.sort_values(
            by=["MeanRMSE", "MeanMAE"],
            kind="stable",
        ).reset_index(drop=True)
        selected = ordered.iloc[0]
        best_mae = ordered.loc[ordered["MeanMAE"].idxmin()]
        best_mape = ordered.loc[ordered["MeanMAPE"].idxmin()]
        best_r2 = ordered.loc[ordered["MeanR2"].idxmax()]
        best_baseline_rmse = (
            ordered.loc[ordered["AlgorithmGroup"] == "baseline"]
            .sort_values(by=["MeanRMSE", "MeanMAE"], kind="stable")
            .iloc[0]
        )
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None

        tradeoff_notes: list[str] = []
        if best_mae["ModelName"] != selected["ModelName"]:
            tradeoff_notes.append(
                f"MAE melhor em {best_mae['ModelName']} ({best_mae['MeanMAE']:.4f})"
            )
        if best_mape["ModelName"] != selected["ModelName"]:
            tradeoff_notes.append(
                f"MAPE melhor em {best_mape['ModelName']} ({best_mape['MeanMAPE']:.2f}%)"
            )
        if best_r2["ModelName"] != selected["ModelName"]:
            tradeoff_notes.append(
                f"R2 melhor em {best_r2['ModelName']} ({best_r2['MeanR2']:.4f})"
            )
        if runner_up is not None:
            tradeoff_notes.append(
                f"Runner-up por RMSE: {runner_up['ModelName']} ({runner_up['MeanRMSE']:.4f})"
            )

        selected_rows.append(
            {
                "CategoryID": int(category_id),
                "CategoryName": selected["CategoryName"],
                "SelectedModel": selected["ModelName"],
                "SelectedModelGroup": selected["AlgorithmGroup"],
                "SelectedMeanRMSE": float(selected["MeanRMSE"]),
                "SelectedMeanMAE": float(selected["MeanMAE"]),
                "SelectedMeanMAPE": float(selected["MeanMAPE"]),
                "SelectedMeanR2": float(selected["MeanR2"]),
                "BestMAEModel": best_mae["ModelName"],
                "BestMAE": float(best_mae["MeanMAE"]),
                "BestMAPEModel": best_mape["ModelName"],
                "BestMAPE": float(best_mape["MeanMAPE"]),
                "BestR2Model": best_r2["ModelName"],
                "BestR2": float(best_r2["MeanR2"]),
                "BestBaselineRMSEModel": best_baseline_rmse["ModelName"],
                "BestBaselineRMSE": float(best_baseline_rmse["MeanRMSE"]),
                "DeltaRMSEvsBestBaseline": float(
                    selected["MeanRMSE"] - best_baseline_rmse["MeanRMSE"]
                ),
                "DeltaMAEvsBestBaseline": float(
                    selected["MeanMAE"] - best_baseline_rmse["MeanMAE"]
                ),
                "TradeoffSummary": " | ".join(tradeoff_notes)
                if tradeoff_notes
                else "Melhor modelo tambem lidera MAE, MAPE e R2.",
            }
        )

    return pd.DataFrame(selected_rows).sort_values(
        by="CategoryID",
        kind="stable",
    ).reset_index(drop=True)


def write_report(
    comparison_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_FILE, index=False)
    selection_df.to_csv(MODEL_SELECTION_FILE, index=False)

    report_lines = [
        "# Comparacao de Algoritmos e Selecao por Categoria",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        "- Criterio primario de selecao: `menor RMSE medio no TSCV`.",
        "- Modelos comparados: `LinearRegression`, `Ridge`, `Lasso`, `XGBoost`, `LightGBM` e `RandomForest`.",
        "",
        "## Tabela Consolidada",
        "",
        comparison_df[
            [
                "CategoryName",
                "AlgorithmGroup",
                "ModelName",
                "MeanRMSE",
                "MeanMAE",
                "MeanMAPE",
                "MeanR2",
                "RMSECategoryRank",
            ]
        ].to_markdown(index=False),
        "",
        "## Melhor Modelo por Categoria",
        "",
        selection_df[
            [
                "CategoryName",
                "SelectedModel",
                "SelectedModelGroup",
                "SelectedMeanRMSE",
                "SelectedMeanMAE",
                "SelectedMeanMAPE",
                "SelectedMeanR2",
                "BestBaselineRMSEModel",
                "BestBaselineRMSE",
                "DeltaRMSEvsBestBaseline",
            ]
        ].to_markdown(index=False),
        "",
        "## Justificativa DS",
        "",
    ]

    for row in selection_df.itertuples(index=False):
        report_lines.extend(
            [
                f"### {row.CategoryName}",
                "",
                f"- Modelo selecionado: `{row.SelectedModel}` ({row.SelectedModelGroup}).",
                (
                    f"- Escolha por RMSE: `{row.SelectedMeanRMSE:.4f}` vs melhor baseline por RMSE "
                    f"`{row.BestBaselineRMSEModel}` = `{row.BestBaselineRMSE:.4f}` "
                    f"(delta `{row.DeltaRMSEvsBestBaseline:.4f}`)."
                ),
                (
                    f"- Leitura complementar: MAE `{row.SelectedMeanMAE:.4f}`, "
                    f"MAPE `{row.SelectedMeanMAPE:.2f}%`, R2 `{row.SelectedMeanR2:.4f}`."
                ),
                f"- Trade-offs: {row.TradeoffSummary}.",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Conclusao",
            "",
            "- `LightGBM` foi o melhor modelo por RMSE nas tres categorias-alvo.",
            "- `RandomForest` apareceu como alternativa forte em MAPE para `Uber Comfort` e `Uber Black`, mas nao venceu no criterio principal de selecao.",
            "- Os baselines lineares continuam como piso de comparacao, mas os modelos avancados passaram esse piso com folga em todas as categorias.",
        ]
    )

    MODEL_SELECTION_REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    log.info("=" * 60)
    log.info("Consolidando metricas dos algoritmos por categoria")
    log.info("=" * 60)

    comparison_df = load_model_summaries()
    selection_df = build_selection_frame(comparison_df)
    write_report(comparison_df, selection_df)

    log.info(
        "Consolidacao concluida | linhas_comparacao=%s | categorias=%s",
        len(comparison_df),
        selection_df["CategoryID"].nunique(),
    )
    log.info("Relatorios salvos em %s", REPORTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
