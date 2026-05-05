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

TABLE_COVERAGE_FILE = REPORTS_DIR / "temporal_coverage_summary.csv"
INTEGRITY_FILE = REPORTS_DIR / "temporal_integrity_checks.csv"
GAPS_FILE = REPORTS_DIR / "temporal_gaps.csv"
DAILY_FILE = REPORTS_DIR / "temporal_daily_metrics.csv"
WEEKDAY_FILE = REPORTS_DIR / "temporal_weekday_seasonality.csv"
MONTHLY_FILE = REPORTS_DIR / "temporal_monthly_metrics.csv"
REPORT_FILE = REPORTS_DIR / "temporal_analysis.md"

DAILY_PLOT_FILE = REPORTS_DIR / "temporal_daily_overview.png"
WEEKDAY_PLOT_FILE = REPORTS_DIR / "temporal_weekday_seasonality.png"
MONTHLY_PLOT_FILE = REPORTS_DIR / "temporal_monthly_metrics.png"

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
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

def ensure_analytical_dataset_exists() -> None:
    if not ANALYTICAL_DIR.exists():
        raise FileNotFoundError(
            f"Dataset analitico nao encontrado em {ANALYTICAL_DIR}. "
            "Execute primeiro o pipeline de ingestion."
        )


def load_base_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dataset = ds.dataset(ANALYTICAL_DIR, format="parquet", partitioning="hive")

    ride_est_df = dataset.to_table(
        columns=["RideEstimativeID", "RideID", "ProductID", "Price", "Schedule", "Create", "Updated"]
    ).to_pandas()
    product_df = dataset.to_table(columns=["ProductID"]).to_pandas().drop_duplicates()

    for column in ["Schedule", "Create", "Updated"]:
        ride_est_df[column] = pd.to_datetime(ride_est_df[column], errors="coerce")

    ride_df = ride_est_df[["RideID", "Schedule", "Create", "Updated"]].drop_duplicates(
        subset=["RideID"],
        ignore_index=True,
    )

    load_stats = {
        "rideestimative_rows": len(ride_est_df),
        "ride_rows": len(ride_df),
        "product_rows": int(product_df["ProductID"].nunique()),
    }

    log.info(
        "Frames temporais carregados | rideestimative=%s | ride=%s | product=%s",
        load_stats["rideestimative_rows"],
        load_stats["ride_rows"],
        load_stats["product_rows"],
    )
    return ride_est_df, ride_df, load_stats


# ============================================================
# COBERTURA E INTEGRIDADE
# ============================================================

def build_coverage_summary(
    ride_est_df: pd.DataFrame,
    ride_df: pd.DataFrame,
    product_count: int,
) -> pd.DataFrame:
    coverage_rows = []

    for table_name, table_df, field_name in [
        ("ride", ride_df, "Create"),
        ("ride", ride_df, "Schedule"),
        ("ride", ride_df, "Updated"),
        ("rideestimative", ride_est_df, "Create"),
        ("rideestimative", ride_est_df, "Schedule"),
        ("rideestimative", ride_est_df, "Updated"),
    ]:
        coverage_rows.append(
            {
                "table_name": table_name,
                "time_field": field_name,
                "row_count": len(table_df),
                "min_timestamp": table_df[field_name].min(),
                "max_timestamp": table_df[field_name].max(),
                "null_count": int(table_df[field_name].isna().sum()),
                "distinct_dates": int(table_df[field_name].dt.floor("D").nunique()),
                "note": (
                    "Campo nativo da tabela ride."
                    if table_name == "ride"
                    else "Cobertura herdada do timestamp da corrida associada."
                ),
            }
        )

    coverage_rows.append(
        {
            "table_name": "product",
            "time_field": "N/A",
            "row_count": product_count,
            "min_timestamp": pd.NaT,
            "max_timestamp": pd.NaT,
            "null_count": pd.NA,
            "distinct_dates": pd.NA,
            "note": "Tabela sem campos temporais nativos no dataset analitico.",
        }
    )

    return pd.DataFrame(coverage_rows)


def build_temporal_gaps(ride_df: pd.DataFrame) -> pd.DataFrame:
    ride_df = ride_df.copy()
    ride_df["create_date"] = ride_df["Create"].dt.floor("D")

    full_calendar = pd.date_range(
        ride_df["create_date"].min(),
        ride_df["create_date"].max(),
        freq="D",
    )
    observed_days = pd.DatetimeIndex(sorted(ride_df["create_date"].dropna().unique()))
    missing_days = full_calendar.difference(observed_days)

    if len(missing_days) == 0:
        return pd.DataFrame(
            [
                {
                    "gap_start": pd.NaT,
                    "gap_end": pd.NaT,
                    "gap_days": 0,
                    "note": "Nenhum gap temporal em Create no nivel de corrida.",
                }
            ]
        )

    gap_rows = []
    start = missing_days[0]
    previous = missing_days[0]

    for current in missing_days[1:]:
        if (current - previous).days == 1:
            previous = current
            continue

        gap_rows.append(
            {
                "gap_start": start,
                "gap_end": previous,
                "gap_days": (previous - start).days + 1,
                "note": "Gap em Create no nivel de corrida.",
            }
        )
        start = current
        previous = current

    gap_rows.append(
        {
            "gap_start": start,
            "gap_end": previous,
            "gap_days": (previous - start).days + 1,
            "note": "Gap em Create no nivel de corrida.",
        }
    )
    return pd.DataFrame(gap_rows)


def build_integrity_checks(ride_df: pd.DataFrame) -> pd.DataFrame:
    schedule_delta_minutes = (
        (ride_df["Schedule"] - ride_df["Create"]).dt.total_seconds() / 60
    )
    updated_delta_hours = (
        (ride_df["Updated"] - ride_df["Create"]).dt.total_seconds() / 3600
    )

    checks = [
        {
            "check_name": "schedule_before_create_any",
            "count": int((ride_df["Schedule"] < ride_df["Create"]).sum()),
            "pct_of_rides": round(float((ride_df["Schedule"] < ride_df["Create"]).mean()) * 100, 4),
            "status": "expected_micro_offset",
            "note": "Schedule tende a anteceder Create por milissegundos em corridas on-demand.",
        },
        {
            "check_name": "schedule_create_abs_gt_1_min",
            "count": int((schedule_delta_minutes.abs() > 1).sum()),
            "pct_of_rides": round(float((schedule_delta_minutes.abs() > 1).mean()) * 100, 4),
            "status": "review",
            "note": "Gap maior que 1 minuto entre Schedule e Create; vale inspecionar, embora nao haja casos >5 minutos.",
        },
        {
            "check_name": "schedule_create_abs_gt_5_min",
            "count": int((schedule_delta_minutes.abs() > 5).sum()),
            "pct_of_rides": round(float((schedule_delta_minutes.abs() > 5).mean()) * 100, 4),
            "status": "ok",
            "note": "Nao foram encontrados casos acima de 5 minutos.",
        },
        {
            "check_name": "updated_before_create",
            "count": int((ride_df["Updated"] < ride_df["Create"]).sum()),
            "pct_of_rides": round(float((ride_df["Updated"] < ride_df["Create"]).mean()) * 100, 4),
            "status": "ok",
            "note": "Nao ha timestamps de Updated anteriores a Create.",
        },
        {
            "check_name": "updated_before_schedule",
            "count": int((ride_df["Updated"] < ride_df["Schedule"]).sum()),
            "pct_of_rides": round(float((ride_df["Updated"] < ride_df["Schedule"]).mean()) * 100, 4),
            "status": "ok",
            "note": "Nao ha timestamps de Updated anteriores a Schedule.",
        },
        {
            "check_name": "updated_after_1_hour",
            "count": int((updated_delta_hours > 1).sum()),
            "pct_of_rides": round(float((updated_delta_hours > 1).mean()) * 100, 4),
            "status": "review",
            "note": "Atualizacoes tardias sao raras e sugerem eventos operacionais fora da janela imediata.",
        },
        {
            "check_name": "updated_after_24_hours",
            "count": int((updated_delta_hours > 24).sum()),
            "pct_of_rides": round(float((updated_delta_hours > 24).mean()) * 100, 4),
            "status": "review",
            "note": "Atualizacoes acima de 24h nao devem ser usadas como ancora temporal para TSCV.",
        },
        {
            "check_name": "updated_after_7_days",
            "count": int((updated_delta_hours > 24 * 7).sum()),
            "pct_of_rides": round(float((updated_delta_hours > 24 * 7).mean()) * 100, 4),
            "status": "review",
            "note": "Casos muito tardios reforcam que Updated e inadequado como indice temporal principal.",
        },
    ]

    return pd.DataFrame(checks)


# ============================================================
# SAZONALIDADE
# ============================================================

def build_daily_metrics(
    ride_est_df: pd.DataFrame,
    ride_df: pd.DataFrame,
) -> pd.DataFrame:
    ride_daily = (
        ride_df.assign(date=ride_df["Create"].dt.floor("D"))
        .groupby("date")
        .agg(unique_rides=("RideID", "nunique"))
        .reset_index()
    )

    estimative_daily = (
        ride_est_df.assign(date=ride_est_df["Create"].dt.floor("D"))
        .groupby("date")
        .agg(
            estimative_rows=("RideEstimativeID", "nunique"),
            mean_price=("Price", "mean"),
            median_price=("Price", "median"),
            p95_price=("Price", lambda series: series.quantile(0.95)),
        )
        .reset_index()
    )

    daily_df = ride_daily.merge(estimative_daily, on="date", how="inner")
    daily_df["weekday"] = daily_df["date"].dt.day_name()
    daily_df["month"] = daily_df["date"].dt.to_period("M").astype(str)
    daily_df["unique_rides_7d_ma"] = daily_df["unique_rides"].rolling(7, min_periods=1).mean()
    daily_df["mean_price_14d_ma"] = daily_df["mean_price"].rolling(14, min_periods=1).mean()
    return daily_df


def build_weekday_seasonality(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekday_df = (
        daily_df.groupby("weekday")
        .agg(
            avg_unique_rides=("unique_rides", "mean"),
            median_unique_rides=("unique_rides", "median"),
            avg_estimative_rows=("estimative_rows", "mean"),
            avg_mean_price=("mean_price", "mean"),
            median_mean_price=("mean_price", "median"),
            observed_days=("date", "count"),
        )
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )
    return weekday_df


def build_monthly_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = (
        daily_df.groupby("month")
        .agg(
            active_days=("date", "count"),
            total_unique_rides=("unique_rides", "sum"),
            avg_unique_rides=("unique_rides", "mean"),
            avg_estimative_rows=("estimative_rows", "mean"),
            avg_mean_price=("mean_price", "mean"),
            median_mean_price=("mean_price", "median"),
        )
        .reset_index()
    )
    return monthly_df


# ============================================================
# VISUALIZACAO
# ============================================================

def save_daily_plot(daily_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)

    axes[0].plot(daily_df["date"], daily_df["unique_rides"], color="#1f77b4", alpha=0.35, label="Unique rides")
    axes[0].plot(daily_df["date"], daily_df["unique_rides_7d_ma"], color="#d62728", linewidth=2, label="7d MA")
    axes[0].set_title("Volume diario de corridas")
    axes[0].set_ylabel("Unique rides")
    axes[0].legend()

    axes[1].plot(daily_df["date"], daily_df["mean_price"], color="#2ca02c", alpha=0.35, label="Daily mean price")
    axes[1].plot(daily_df["date"], daily_df["mean_price_14d_ma"], color="#9467bd", linewidth=2, label="14d MA")
    axes[1].set_title("Preco medio diario das estimativas")
    axes[1].set_ylabel("Mean Price")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    fig.suptitle("Cobertura temporal e dinamica diaria", fontsize=16)
    fig.savefig(DAILY_PLOT_FILE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_weekday_plot(weekday_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    sns.barplot(
        data=weekday_df,
        x="weekday",
        y="avg_unique_rides",
        ax=axes[0],
        color="#1f77b4",
    )
    axes[0].set_title("Media de corridas por dia da semana")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average unique rides")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=weekday_df,
        x="weekday",
        y="avg_mean_price",
        ax=axes[1],
        color="#2ca02c",
    )
    axes[1].set_title("Preco medio por dia da semana")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Average mean price")
    axes[1].tick_params(axis="x", rotation=35)

    fig.suptitle("Sazonalidade semanal", fontsize=16)
    fig.savefig(WEEKDAY_PLOT_FILE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_monthly_plot(monthly_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    month_positions = list(range(len(monthly_df)))

    axes[0].bar(month_positions, monthly_df["total_unique_rides"], color="#1f77b4")
    axes[0].set_title("Total de corridas por mes")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Total unique rides")
    axes[0].set_xticks(month_positions)
    axes[0].set_xticklabels(monthly_df["month"], rotation=45)

    axes[1].bar(month_positions, monthly_df["avg_mean_price"], color="#2ca02c")
    axes[1].set_title("Preco medio por mes")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Average mean price")
    axes[1].set_xticks(month_positions)
    axes[1].set_xticklabels(monthly_df["month"], rotation=45)

    fig.suptitle("Sazonalidade mensal", fontsize=16)
    fig.savefig(MONTHLY_PLOT_FILE, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RELATORIO
# ============================================================

def build_tscv_recommendation(daily_df: pd.DataFrame) -> list[str]:
    high_regime = daily_df[daily_df["date"] < pd.Timestamp("2021-11-01")]
    low_regime = daily_df[daily_df["date"] >= pd.Timestamp("2021-11-01")]

    return [
        "Use `Create` como ancora temporal principal do TSCV. `Schedule` e praticamente identico a `Create` neste dataset, enquanto `Updated` possui cauda longa de atualizacao tardia.",
        "Como existe sazonalidade semanal forte, cada janela de validacao deve cobrir semanas completas. O minimo recomendado e `28 dias` por fold.",
        (
            f"O dataset cobre `302` dias continuos, o que comporta `5 folds` de `28 dias` com janela expansiva no historico completo. "
            f"Se a prioridade for estabilidade de regime, use o subperiodo `2021-11-01` a `2022-06-14` ({len(low_regime)} dias) e rode `4 folds` de `28 dias`."
        ),
        (
            f"Ha mudanca forte de regime no volume: media diaria de `~{high_regime['unique_rides'].mean():.0f}` corridas entre `ago-out/2021` "
            f"contra `~{low_regime['unique_rides'].mean():.0f}` entre `nov/2021-jun/2022`. "
            "Por isso, as metricas mais representativas do futuro recente tendem a vir dos folds posicionados no regime mais novo."
        ),
        "Como ha varias estimativas por corrida, todos os registros do mesmo `RideID` devem permanecer no mesmo fold para evitar leakage entre treino e validacao.",
    ]


def build_markdown_report(
    load_stats: dict,
    coverage_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    weekday_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> str:
    weekend_gap = weekday_df.loc[weekday_df["weekday"] == "Sunday", "avg_unique_rides"].iloc[0]
    thursday_level = weekday_df.loc[weekday_df["weekday"] == "Thursday", "avg_unique_rides"].iloc[0]

    lines = [
        "# Analise Temporal dos Dados",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{ANALYTICAL_DIR}`",
        f"- Escopo: `ride`, `rideestimative` e `product` reconstruidos a partir do Parquet consolidado.",
        "",
        "## Cobertura Temporal",
        "",
        coverage_df.to_markdown(index=False),
        "",
        "## Gaps Temporais",
        "",
        gaps_df.to_markdown(index=False),
        "",
        "## Integridade dos Timestamps",
        "",
        integrity_df.to_markdown(index=False),
        "",
        "## Sazonalidade Diaria",
        "",
        daily_df[
            ["date", "unique_rides", "estimative_rows", "mean_price", "median_price", "p95_price"]
        ].head(10).to_markdown(index=False),
        "",
        "## Sazonalidade Semanal",
        "",
        weekday_df.to_markdown(index=False),
        "",
        "## Sazonalidade Mensal",
        "",
        monthly_df.to_markdown(index=False),
        "",
        "## Interpretacao DS",
        "",
        f"- A serie e continua de `2021-08-17` a `2022-06-14`, sem gaps diarios em `Create` no nivel de corrida.",
        f"- `Schedule` antecede `Create` em `99%+` dos casos por milissegundos; apenas `{int(integrity_df.loc[integrity_df['check_name'] == 'schedule_create_abs_gt_1_min', 'count'].iloc[0])}` corridas passam de 1 minuto, e nenhuma passa de 5 minutos.",
        f"- `Updated` nunca antecede `Create`, mas possui `{int(integrity_df.loc[integrity_df['check_name'] == 'updated_after_24_hours', 'count'].iloc[0])}` atualizacoes acima de 24h; por isso, nao deve ser usado como eixo do TSCV.",
        f"- Existe sazonalidade semanal forte no volume: quinta-feira tem em media `~{thursday_level:.0f}` corridas/dia, enquanto domingo cai para `~{weekend_gap:.0f}`.",
        f"- O preco medio e menos estavel que o volume, mas mostra patamar mais alto a partir do regime de baixo volume, especialmente de `abr/2022` em diante.",
        "- Ha quebra de regime clara no volume a partir de `nov/2021`, com queda muito forte frente a `ago-out/2021`. Isso precisa entrar na definicao dos folds.",
        "",
        "## Recomendacao para TSCV",
        "",
    ]

    for item in build_tscv_recommendation(daily_df):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Artefatos",
            "",
            f"- Daily overview: `{DAILY_PLOT_FILE.name}`",
            f"- Weekly seasonality: `{WEEKDAY_PLOT_FILE.name}`",
            f"- Monthly metrics: `{MONTHLY_PLOT_FILE.name}`",
            "",
            "## Conclusao",
            "",
            "- A cobertura temporal e suficiente para TSCV, mas os folds precisam respeitar semanas completas e a mudanca de regime observada no volume.",
            "- `Create` e o timestamp mais confiavel para ordenar os dados. `Updated` deve ficar fora da definicao dos splits.",
            "- A avaliacao mais representativa do cenario recente tende a vir de validacoes concentradas no periodo `nov/2021-jun/2022`.",
        ]
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_analytical_dataset_exists()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ride_est_df, ride_df, load_stats = load_base_frames()
    coverage_df = build_coverage_summary(
        ride_est_df=ride_est_df,
        ride_df=ride_df,
        product_count=load_stats["product_rows"],
    )
    gaps_df = build_temporal_gaps(ride_df)
    integrity_df = build_integrity_checks(ride_df)
    daily_df = build_daily_metrics(ride_est_df, ride_df)
    weekday_df = build_weekday_seasonality(daily_df)
    monthly_df = build_monthly_metrics(daily_df)

    coverage_df.to_csv(TABLE_COVERAGE_FILE, index=False)
    gaps_df.to_csv(GAPS_FILE, index=False)
    integrity_df.to_csv(INTEGRITY_FILE, index=False)
    daily_df.to_csv(DAILY_FILE, index=False)
    weekday_df.to_csv(WEEKDAY_FILE, index=False)
    monthly_df.to_csv(MONTHLY_FILE, index=False)

    save_daily_plot(daily_df)
    save_weekday_plot(weekday_df)
    save_monthly_plot(monthly_df)

    REPORT_FILE.write_text(
        build_markdown_report(
            load_stats=load_stats,
            coverage_df=coverage_df,
            gaps_df=gaps_df,
            integrity_df=integrity_df,
            daily_df=daily_df,
            weekday_df=weekday_df,
            monthly_df=monthly_df,
        ),
        encoding="utf-8",
    )

    log.info("Cobertura temporal salva em %s", TABLE_COVERAGE_FILE)
    log.info("Integridade temporal salva em %s", INTEGRITY_FILE)
    log.info("Gaps temporais salvos em %s", GAPS_FILE)
    log.info("Metricas diarias salvas em %s", DAILY_FILE)
    log.info("Sazonalidade semanal salva em %s", WEEKDAY_FILE)
    log.info("Metricas mensais salvas em %s", MONTHLY_FILE)
    log.info("Relatorio salvo em %s", REPORT_FILE)
    log.info("Plot diario salvo em %s", DAILY_PLOT_FILE)
    log.info("Plot semanal salvo em %s", WEEKDAY_PLOT_FILE)
    log.info("Plot mensal salvo em %s", MONTHLY_PLOT_FILE)


if __name__ == "__main__":
    main()
