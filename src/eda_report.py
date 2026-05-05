from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / ".cache" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))

import sweetviz as sv


# ============================================================
# CONFIGURACAO
# ============================================================

RUN_AT = datetime.now().isoformat()
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SOURCE_DIR = DATA_DIR / "analytical"
REPORT_FILE = REPORTS_DIR / "eda_report.html"
SUMMARY_FILE = REPORTS_DIR / "eda_report_summary.md"
NOTEBOOK_FILE = NOTEBOOKS_DIR / "eda_report_notes.ipynb"

TARGET_CATEGORIES = {
    2: "UberX",
    9: "Uber Comfort",
    4: "Uber Black",
}
TARGET_CATEGORY_IDS = list(TARGET_CATEGORIES.keys())
MAX_ROWS_PER_CATEGORY = 100_000
RANDOM_STATE = 42

SOURCE_COLUMNS = [
    "CategoryID",
    "ProductID",
    "Price",
    "WaitingTime",
    "Fee",
    "FareID",
    "RideStatusID",
    "CompanyID",
    "ProductProviderID",
    "TotalUsers",
    "Create",
    "Schedule",
    "OriginCity",
    "OriginState",
    "DestinationCity",
    "DestinationState",
]

REPORT_COLUMNS = [
    "CategoryName",
    "ProductID",
    "Price",
    "WaitingTime",
    "Fee",
    "TotalUsers",
    "RideStatusID",
    "CompanyID",
    "ProductProviderID",
    "CreateMonth",
    "CreateHour",
    "CreateWeekday",
    "LeadTimeSeconds",
    "FareIDMissing",
    "OriginCity",
    "OriginState",
    "DestinationCity",
    "DestinationState",
]

CATEGORICAL_COLUMNS = [
    "CategoryName",
    "ProductID",
    "RideStatusID",
    "CompanyID",
    "ProductProviderID",
    "CreateMonth",
    "CreateWeekday",
    "OriginCity",
    "OriginState",
    "DestinationCity",
    "DestinationState",
    "FareIDMissing",
]


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
# LEITURA
# ============================================================

def build_filter_expression() -> ds.Expression:
    expression = ds.field("CategoryID") == TARGET_CATEGORY_IDS[0]
    for category_id in TARGET_CATEGORY_IDS[1:]:
        expression = expression | (ds.field("CategoryID") == category_id)
    return expression


def load_filtered_dataframe() -> pd.DataFrame:
    log.info("Lendo Parquet analitico filtrado para as categorias-alvo...")
    dataset = ds.dataset(
        SOURCE_DIR,
        format="parquet",
        partitioning="hive",
    )
    table = dataset.to_table(
        columns=SOURCE_COLUMNS,
        filter=build_filter_expression(),
    )
    df = table.to_pandas()

    numeric_columns = [
        "CategoryID",
        "Price",
        "WaitingTime",
        "Fee",
        "RideStatusID",
        "CompanyID",
        "ProductProviderID",
        "TotalUsers",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["Create", "Schedule"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df["CategoryName"] = df["CategoryID"].map(TARGET_CATEGORIES)
    df["CreateMonth"] = df["Create"].dt.to_period("M").astype("string")
    df["CreateHour"] = df["Create"].dt.hour.astype("Int64")
    df["CreateWeekday"] = df["Create"].dt.day_name().astype("string")
    df["LeadTimeSeconds"] = (
        (df["Create"] - df["Schedule"]).dt.total_seconds().abs()
    )
    df["FareIDMissing"] = (
        df["FareID"].isna()
        | df["FareID"].astype("string").str.strip().eq("")
    )

    fill_with_missing = [
        "ProductID",
        "OriginCity",
        "OriginState",
        "DestinationCity",
        "DestinationState",
        "CreateMonth",
        "CreateWeekday",
    ]
    for column in fill_with_missing:
        df[column] = df[column].astype("string").fillna("MISSING")

    log.info("DataFrame filtrado pronto | linhas=%s | colunas=%s", len(df), len(df.columns))
    return df


def build_report_sample(df: pd.DataFrame) -> pd.DataFrame:
    log.info(
        "Montando amostra estratificada reproducivel | max_por_categoria=%s",
        MAX_ROWS_PER_CATEGORY,
    )
    sampled_frames = []
    for category_id, category_name in TARGET_CATEGORIES.items():
        category_df = df.loc[df["CategoryID"] == category_id, REPORT_COLUMNS].copy()
        sample_size = min(len(category_df), MAX_ROWS_PER_CATEGORY)
        if len(category_df) > sample_size:
            category_df = category_df.sample(
                n=sample_size,
                random_state=RANDOM_STATE,
            )
        sampled_frames.append(category_df)
        log.info(
            "Amostra %s | populacao=%s | amostra=%s",
            category_name,
            int((df["CategoryID"] == category_id).sum()),
            sample_size,
        )

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    for column in CATEGORICAL_COLUMNS:
        if column in sampled_df.columns:
            sampled_df[column] = sampled_df[column].astype("string")
    return sampled_df


# ============================================================
# RELATORIO HTML
# ============================================================

def generate_html_report(sample_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_config = sv.FeatureConfig(
        force_cat=tuple(CATEGORICAL_COLUMNS),
    )
    report = sv.analyze(
        (sample_df, "Analytical Parquet - Target Categories"),
        feat_cfg=feature_config,
        pairwise_analysis="off",
    )
    report.show_html(
        filepath=str(REPORT_FILE),
        open_browser=False,
        layout="vertical",
    )
    log.info("EDA HTML salvo em %s", REPORT_FILE)


# ============================================================
# SUMARIO DS + NOTEBOOK
# ============================================================

def build_summary_payload(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> dict:
    rows_by_category = {
        TARGET_CATEGORIES[category_id]: int((full_df["CategoryID"] == category_id).sum())
        for category_id in TARGET_CATEGORY_IDS
    }
    sample_rows_by_category = {
        category_name: int((sample_df["CategoryName"] == category_name).sum())
        for category_name in TARGET_CATEGORIES.values()
    }

    price_summary = (
        full_df.groupby("CategoryName")["Price"]
        .agg(["mean", "median", "max", "skew"])
        .round(4)
        .to_dict(orient="index")
    )
    quantiles = (
        full_df.groupby("CategoryName")["Price"]
        .quantile([0.95, 0.99])
        .unstack()
        .round(4)
        .rename(columns={0.95: "p95", 0.99: "p99"})
        .to_dict(orient="index")
    )
    waiting_time_p95 = (
        full_df.groupby("CategoryName")["WaitingTime"]
        .quantile(0.95)
        .round(4)
        .to_dict()
    )
    fare_missing_pct = (
        full_df.groupby("CategoryName")["FareIDMissing"]
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )
    weekday_volume = (
        full_df.groupby("CreateWeekday")
        .size()
        .sort_values(ascending=False)
        .head(3)
        .to_dict()
    )

    return {
        "generated_at": RUN_AT,
        "source_dir": str(SOURCE_DIR),
        "report_file": str(REPORT_FILE),
        "notebook_file": str(NOTEBOOK_FILE),
        "rows_filtered": int(len(full_df)),
        "rows_by_category": rows_by_category,
        "sample_rows": int(len(sample_df)),
        "sample_rows_by_category": sample_rows_by_category,
        "date_min": str(full_df["Create"].min()),
        "date_max": str(full_df["Create"].max()),
        "price_summary": price_summary,
        "price_quantiles": quantiles,
        "waiting_time_p95": waiting_time_p95,
        "fare_missing_pct": fare_missing_pct,
        "weekday_volume_top3": weekday_volume,
        "lead_time_over_60s": int((full_df["LeadTimeSeconds"] > 60).sum()),
        "lead_time_over_300s": int((full_df["LeadTimeSeconds"] > 300).sum()),
    }


def write_summary_markdown(summary: dict) -> None:
    summary_lines = [
        "# EDA Consolidado das Categorias-Alvo",
        "",
        f"- Gerado em: `{summary['generated_at']}`",
        f"- Fonte: `{summary['source_dir']}`",
        f"- Relatorio HTML: `{summary['report_file']}`",
        f"- Populacao analisada (categorias 2, 9, 4): `{summary['rows_filtered']}` linhas",
        f"- Amostra Sweetviz: `{summary['sample_rows']}` linhas",
        f"- Cobertura temporal: `{summary['date_min']}` ate `{summary['date_max']}`",
        "",
        "## Cobertura por Categoria",
        "",
        pd.DataFrame(
            [
                {
                    "CategoryName": category_name,
                    "population_rows": summary["rows_by_category"][category_name],
                    "sample_rows": summary["sample_rows_by_category"][category_name],
                }
                for category_name in TARGET_CATEGORIES.values()
            ]
        ).to_markdown(index=False),
        "",
        "## Principais Achados DS",
        "",
        (
            f"- `UberX` concentra o maior volume (`{summary['rows_by_category']['UberX']}` linhas) "
            "e continua com a cauda de `Price` mais longa entre as categorias-alvo."
        ),
        (
            f"- `Uber Black` tem o maior nivel central de preco "
            f"(mediana `{summary['price_summary']['Uber Black']['median']}` e p99 "
            f"`{summary['price_quantiles']['Uber Black']['p99']}`), mesmo com menor volume."
        ),
        (
            f"- `Uber Comfort` fica no meio do caminho em volume e patamar de preco, "
            f"mas mantem assimetria forte (`skew={summary['price_summary']['Uber Comfort']['skew']}`)."
        ),
        (
            f"- O missing de `FareID` permanece alto nas tres categorias "
            f"(`UberX={summary['fare_missing_pct']['UberX']}%`, "
            f"`Uber Comfort={summary['fare_missing_pct']['Uber Comfort']}%`, "
            f"`Uber Black={summary['fare_missing_pct']['Uber Black']}%`)."
        ),
        (
            f"- `WaitingTime` segue concentrado em janelas curtas; o p95 por categoria ficou em "
            f"`UberX={summary['waiting_time_p95']['UberX']}`, "
            f"`Uber Comfort={summary['waiting_time_p95']['Uber Comfort']}` e "
            f"`Uber Black={summary['waiting_time_p95']['Uber Black']}` minutos."
        ),
        (
            f"- O volume semanal segue concentrado em dias uteis; os tres maiores dias no recorte foram "
            f"`{summary['weekday_volume_top3']}`."
        ),
        "",
        "## Observacao de Configuracao",
        "",
        (
            f"- O HTML foi gerado com `sweetviz`, `pairwise_analysis='off'` e amostragem estratificada "
            f"deterministica de ate `{MAX_ROWS_PER_CATEGORY}` linhas por categoria para manter o processo "
            "reproduzivel e o artefato navegavel no volume atual."
        ),
    ]
    SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("Resumo Markdown salvo em %s", SUMMARY_FILE)


def build_notebook_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "# Comentarios DS sobre o `eda_report.html`",
            "",
            (
                "Notebook gerado automaticamente para registrar os principais achados "
                "do relatorio HTML consolidado das categorias `UberX`, `Uber Comfort` e `Uber Black`."
            ),
            "",
            "## Configuracao do Relatorio",
            "",
            f"- Fonte: `{summary['source_dir']}`",
            f"- Populacao filtrada: `{summary['rows_filtered']}` linhas",
            (
                f"- Amostra Sweetviz: `{summary['sample_rows']}` linhas "
                f"({MAX_ROWS_PER_CATEGORY} por categoria, quando disponivel)"
            ),
            (
                "- Parametros: `pairwise_analysis='off'`, `CategoryName` tratada como feature categorial-chave e "
                "features temporais derivadas de `Create`."
            ),
            "",
            "## Principais Achados",
            "",
            (
                f"- `UberX` domina o volume (`{summary['rows_by_category']['UberX']}` linhas) "
                f"e exibe a maior cauda de `Price` (`p99={summary['price_quantiles']['UberX']['p99']}`, "
                f"`max={summary['price_summary']['UberX']['max']}`)."
            ),
            (
                f"- `Uber Black` tem o preco tipico mais alto "
                f"(mediana `{summary['price_summary']['Uber Black']['median']}`), o que reforca "
                "a necessidade de modelagem segmentada ou com interacoes por categoria."
            ),
            (
                f"- `Uber Comfort` continua assimetrico (`skew={summary['price_summary']['Uber Comfort']['skew']}`) "
                "e nao pode ser tratado como distribuicao aproximadamente normal."
            ),
            (
                f"- O missing de `FareID` segue estrutural "
                f"(`UberX={summary['fare_missing_pct']['UberX']}%`, "
                f"`Uber Comfort={summary['fare_missing_pct']['Uber Comfort']}%`, "
                f"`Uber Black={summary['fare_missing_pct']['Uber Black']}%`), "
                "entao a estrategia de sentinel continua correta."
            ),
            (
                f"- `WaitingTime` tem p95 curto "
                f"(`UberX={summary['waiting_time_p95']['UberX']}`, "
                f"`Uber Comfort={summary['waiting_time_p95']['Uber Comfort']}`, "
                f"`Uber Black={summary['waiting_time_p95']['Uber Black']}`), "
                "o que sugere relacao nao linear com `Price` e utilidade para binning."
            ),
            (
                f"- Os gaps entre `Schedule` e `Create` seguem pequenos no recorte: "
                f"`>60s={summary['lead_time_over_60s']}` e `>300s={summary['lead_time_over_300s']}`."
            ),
            "",
            "## Encaminhamento",
            "",
            (
                "- Usar o HTML como visao navegavel consolidada e manter as regras de tratamento "
                "ja documentadas na camada curada antes do treino."
            ),
            (
                "- Priorizar features por categoria, interacoes temporais e transformacoes robustas "
                "para o target devido a assimetria residual."
            ),
        ]
    )


def write_notebook(summary: dict) -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": build_notebook_markdown(summary).splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_FILE.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Notebook de comentarios salvo em %s", NOTEBOOK_FILE)


# ============================================================
# MAIN
# ============================================================

def main() -> dict:
    log.info("=" * 60)
    log.info("Gerando EDA HTML consolidado")
    log.info("=" * 60)

    full_df = load_filtered_dataframe()
    sample_df = build_report_sample(full_df)
    generate_html_report(sample_df)

    summary = build_summary_payload(full_df, sample_df)
    write_summary_markdown(summary)
    write_notebook(summary)

    log.info("=" * 60)
    log.info("EDA HTML finalizado com sucesso")
    log.info("=" * 60)
    return summary


if __name__ == "__main__":
    main()
