import os
import logging
import sys
from datetime import datetime
from pathlib import Path

BOOTSTRAP_BASE_DIR = Path(__file__).resolve().parent.parent
MPLCONFIG_DIR = BOOTSTRAP_BASE_DIR / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds
import seaborn as sns

# ============================================================
# CONFIGURACAO
# ============================================================

BASE_DIR = Path(__file__).parent.parent
ANALYTICAL_DIR = BASE_DIR / "data" / "analytical"
REPORTS_DIR = BASE_DIR / "reports"

SUMMARY_FILE = REPORTS_DIR / "target_price_by_category_summary.csv"
PERCENTILES_FILE = REPORTS_DIR / "target_price_by_category_percentiles.csv"
REPORT_FILE = REPORTS_DIR / "target_price_by_category.md"
HISTOGRAM_FILE = REPORTS_DIR / "target_price_histograms.png"
BOXPLOT_FILE = REPORTS_DIR / "target_price_boxplots.png"

CATEGORY_MAP = {
    2: "UberX",
    9: "Uber Comfort",
    4: "Uber Black",
}

PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
PERCENTILE_LABELS = ["p1", "p5", "p25", "p50", "p75", "p90", "p95", "p99", "p99_5", "p99_9"]


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
# LEITURA FILTRADA
# ============================================================

def ensure_analytical_dataset_exists() -> None:
    if not ANALYTICAL_DIR.exists():
        raise FileNotFoundError(
            f"Dataset analitico nao encontrado em {ANALYTICAL_DIR}. "
            "Execute primeiro o pipeline de ingestion."
        )


def load_price_series(dataset: ds.Dataset, category_id: int) -> tuple[pd.Series, dict]:
    filter_expression = ds.field("CategoryID") == category_id
    fragments = list(dataset.get_fragments(filter=filter_expression))
    table = dataset.to_table(columns=["Price"], filter=filter_expression)
    series = table["Price"].to_pandas().dropna().astype("float64")

    read_stats = {
        "category_id": category_id,
        "category_name": CATEGORY_MAP[category_id],
        "fragments_scanned": len(fragments),
        "rows_loaded": len(series),
    }

    log.info(
        "%s | category_id=%s | fragments=%s | rows=%s",
        read_stats["category_name"],
        category_id,
        read_stats["fragments_scanned"],
        read_stats["rows_loaded"],
    )
    return series, read_stats


# ============================================================
# ESTATISTICAS
# ============================================================

def compute_category_statistics(series: pd.Series, read_stats: dict) -> tuple[dict, dict]:
    percentiles = series.quantile(PERCENTILES)
    q1 = float(percentiles.loc[0.25])
    q3 = float(percentiles.loc[0.75])
    iqr = q3 - q1

    mild_lower = q1 - 1.5 * iqr
    mild_upper = q3 + 1.5 * iqr
    extreme_lower = q1 - 3.0 * iqr
    extreme_upper = q3 + 3.0 * iqr

    mild_outliers = series[(series < mild_lower) | (series > mild_upper)]
    extreme_outliers = series[(series < extreme_lower) | (series > extreme_upper)]

    summary = {
        "CategoryID": read_stats["category_id"],
        "CategoryName": read_stats["category_name"],
        "Rows": len(series),
        "FragmentsScanned": read_stats["fragments_scanned"],
        "Mean": round(float(series.mean()), 4),
        "Median": round(float(series.median()), 4),
        "Std": round(float(series.std()), 4),
        "Min": round(float(series.min()), 4),
        "Max": round(float(series.max()), 4),
        "Skewness": round(float(series.skew()), 4),
        "IQR": round(iqr, 4),
        "MildOutlierLower": round(mild_lower, 4),
        "MildOutlierUpper": round(mild_upper, 4),
        "ExtremeOutlierLower": round(extreme_lower, 4),
        "ExtremeOutlierUpper": round(extreme_upper, 4),
        "MildOutlierCount": int(mild_outliers.shape[0]),
        "MildOutlierPct": round((mild_outliers.shape[0] / len(series)) * 100, 2),
        "ExtremeOutlierCount": int(extreme_outliers.shape[0]),
        "ExtremeOutlierPct": round((extreme_outliers.shape[0] / len(series)) * 100, 2),
        "ZeroPriceCount": int((series == 0).sum()),
        "ZeroPricePct": round(((series == 0).sum() / len(series)) * 100, 2),
    }

    percentile_row = {
        "CategoryID": read_stats["category_id"],
        "CategoryName": read_stats["category_name"],
    }
    percentile_row.update(
        {
            label: round(float(percentiles.loc[quantile]), 4)
            for quantile, label in zip(PERCENTILES, PERCENTILE_LABELS)
        }
    )

    return summary, percentile_row


# ============================================================
# GRAFICOS
# ============================================================

def build_histograms(category_frames: list[pd.DataFrame], summary_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for axis, category_frame in zip(axes, category_frames):
        category_name = category_frame["CategoryName"].iloc[0]
        prices = category_frame["Price"]
        summary_row = summary_df.loc[summary_df["CategoryName"] == category_name].iloc[0]
        x_limit = float(prices.quantile(0.995))

        sns.histplot(
            prices,
            bins=80,
            ax=axis,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.2,
            alpha=0.9,
        )
        axis.set_xlim(0, x_limit)
        axis.set_yscale("log")
        axis.axvline(summary_row["Median"], color="#d62728", linestyle="--", linewidth=1.5, label="Median")
        axis.axvline(summary_row["Mean"], color="#2ca02c", linestyle=":", linewidth=1.5, label="Mean")
        axis.set_title(f"{category_name}\nHistograma ate p99.5")
        axis.set_xlabel("Price")
        axis.set_ylabel("Frequency (log)")
        axis.legend()

    fig.suptitle("Distribuicao de Price por categoria", fontsize=16)
    fig.savefig(HISTOGRAM_FILE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_boxplots(category_frames: list[pd.DataFrame], summary_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    for axis, category_frame in zip(axes, category_frames):
        category_name = category_frame["CategoryName"].iloc[0]
        summary_row = summary_df.loc[summary_df["CategoryName"] == category_name].iloc[0]
        sns.boxplot(
            x=category_frame["Price"],
            ax=axis,
            color="#9ecae1",
            showfliers=False,
        )
        axis.set_title(
            f"{category_name}\nSem fliers visuais | extremos={summary_row['ExtremeOutlierPct']}%"
        )
        axis.set_xlabel("Price")

    fig.suptitle("Boxplots de Price por categoria", fontsize=16)
    fig.savefig(BOXPLOT_FILE, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RELATORIO
# ============================================================

def build_markdown_report(summary_df: pd.DataFrame, percentile_df: pd.DataFrame) -> str:
    lines = [
        "# Analise do Target Price por Categoria",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{ANALYTICAL_DIR}`",
        "- Leitura feita diretamente do Parquet particionado por `CategoryID`, com filtro nas categorias `2`, `9` e `4`.",
        "- Categorias analisadas: `UberX (2)`, `Uber Comfort (9)` e `Uber Black (4)`.",
        "",
        "## Resumo Estatistico",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Percentis",
        "",
        percentile_df.to_markdown(index=False),
        "",
        "## Interpretacao DS",
        "",
    ]

    for _, row in summary_df.iterrows():
        asymmetry_text = "fortemente assimetrica a direita"
        if row["Skewness"] < 1:
            asymmetry_text = "moderadamente assimetrica a direita"
        if row["Skewness"] < 0.5:
            asymmetry_text = "quase simetrica"

        treatment_note = "Recomenda-se tratamento robusto antes do treino."
        if row["ExtremeOutlierPct"] < 0.5 and row["Skewness"] < 1.0:
            treatment_note = "Tratamento pode ser leve, com foco maior em robustez do modelo."

        lines.extend(
            [
                f"### {row['CategoryName']}",
                "",
                f"- Mediana `{row['Median']}` | media `{row['Mean']}` | desvio `{row['Std']}`.",
                f"- Assimetria `{row['Skewness']}`: distribuicao {asymmetry_text}.",
                f"- p95 `{percentile_df.loc[percentile_df['CategoryName'] == row['CategoryName'], 'p95'].iloc[0]}` | "
                f"p99 `{percentile_df.loc[percentile_df['CategoryName'] == row['CategoryName'], 'p99'].iloc[0]}` | "
                f"max `{row['Max']}`.",
                f"- Outliers extremos (regra 3*IQR): `{row['ExtremeOutlierCount']}` linhas (`{row['ExtremeOutlierPct']}%`) acima de `{row['ExtremeOutlierUpper']}`.",
                f"- {treatment_note}",
                "",
            ]
        )

    lines.extend(
        [
            "## Graficos",
            "",
            f"- Histograms: `{HISTOGRAM_FILE.name}`",
            f"- Boxplots: `{BOXPLOT_FILE.name}`",
            "",
            "## Conclusao",
            "",
            "- As tres categorias apresentam cauda a direita e presenca de outliers altos no target.",
            "- Uber Black tende a concentrar valores centrais mais altos e cauda mais longa em termos absolutos.",
            "- Antes do treino, vale testar abordagem robusta para o target, como winsorizacao leve, clipping por regra estatistica ou avaliacao de transformacao log1p em experimentos controlados.",
        ]
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_analytical_dataset_exists()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(ANALYTICAL_DIR, format="parquet", partitioning="hive")
    category_frames = []
    summary_rows = []
    percentile_rows = []

    for category_id in CATEGORY_MAP:
        price_series, read_stats = load_price_series(dataset, category_id)
        summary_row, percentile_row = compute_category_statistics(price_series, read_stats)

        category_frames.append(
            pd.DataFrame(
                {
                    "CategoryID": category_id,
                    "CategoryName": CATEGORY_MAP[category_id],
                    "Price": price_series,
                }
            )
        )
        summary_rows.append(summary_row)
        percentile_rows.append(percentile_row)

    summary_df = pd.DataFrame(summary_rows)
    percentile_df = pd.DataFrame(percentile_rows)

    build_histograms(category_frames, summary_df)
    build_boxplots(category_frames, summary_df)

    summary_df.to_csv(SUMMARY_FILE, index=False)
    percentile_df.to_csv(PERCENTILES_FILE, index=False)
    REPORT_FILE.write_text(
        build_markdown_report(summary_df, percentile_df),
        encoding="utf-8",
    )

    log.info("Resumo salvo em %s", SUMMARY_FILE)
    log.info("Percentis salvos em %s", PERCENTILES_FILE)
    log.info("Relatorio salvo em %s", REPORT_FILE)
    log.info("Histogramas salvos em %s", HISTOGRAM_FILE)
    log.info("Boxplots salvos em %s", BOXPLOT_FILE)


if __name__ == "__main__":
    main()
