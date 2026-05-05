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

TARGET_COLUMN = "Price"
HIGH_CORR_THRESHOLD = 0.85
LOW_TARGET_CORR_THRESHOLD = 0.05
HIGH_NULL_THRESHOLD = 95.0

SELECTED_COLUMNS = [
    "RideEstimativeID",
    "RideID",
    "WaitingTime",
    "Price",
    "Selected",
    "RideReasonSelectedEstimativeID",
    "Fee",
    "RideStatusID",
    "CompanyID",
    "ProviderID",
    "RideProviderID",
    "RidePrice",
    "RideCategoryID",
    "TotalUsers",
    "RideDriverLocationID",
    "ScheduledRide",
    "ProductProviderID",
    "CategoryID",
    "OriginRideAddressID",
    "DestinationRideAddressID",
    "OriginLat",
    "OriginLng",
    "DestinationLat",
    "DestinationLng",
]

NUMERIC_STRING_COLUMNS = [
    "OriginLat",
    "OriginLng",
    "DestinationLat",
    "DestinationLng",
]

EXCLUDED_COLUMNS = {
    "RideEstimativeID": "Identificador unico da estimativa.",
    "RideID": "Identificador da corrida, nao interpretavel como feature numerica.",
    "OriginRideAddressID": "Identificador do endereco de origem.",
    "DestinationRideAddressID": "Identificador do endereco de destino.",
    "ScheduledRide": "Coluna constante no dataset analitico atual.",
    "ProviderID": "Mais de 95% de nulos no dataset consolidado.",
    "RideProviderID": "Mais de 95% de nulos e alta cardinalidade residual.",
    "RideCategoryID": "Mais de 95% de nulos no dataset consolidado.",
    "RideDriverLocationID": "Mais de 95% de nulos e sem interpretacao direta.",
}

LEAKAGE_COLUMNS = {
    "RidePrice": "Preco real da corrida apos a execucao; vazamento explicito.",
    "Selected": "Marcador de escolha da estimativa, conhecido apenas apos a selecao do usuario.",
    "RideReasonSelectedEstimativeID": "Motivo da selecao da estimativa, disponivel apenas apos o evento.",
}

FEATURE_FAMILY = {
    "WaitingTime": "continuous",
    "Price": "target",
    "Selected": "post_event",
    "RideReasonSelectedEstimativeID": "post_event",
    "Fee": "monetary_component",
    "RideStatusID": "categorical_code",
    "CompanyID": "categorical_code",
    "RidePrice": "leakage",
    "TotalUsers": "count",
    "ProductProviderID": "categorical_code",
    "CategoryID": "categorical_code",
    "OriginLat": "spatial_coordinate",
    "OriginLng": "spatial_coordinate",
    "DestinationLat": "spatial_coordinate",
    "DestinationLng": "spatial_coordinate",
}

REPORT_FILE = REPORTS_DIR / "correlation_analysis.md"
CORRELATION_MATRIX_FILE = REPORTS_DIR / "correlation_matrix.csv"
TARGET_CORRELATIONS_FILE = REPORTS_DIR / "target_correlations.csv"
MULTICOLLINEARITY_FILE = REPORTS_DIR / "multicollinearity_pairs.csv"
LOW_SIGNAL_FILE = REPORTS_DIR / "low_target_correlation_candidates.csv"
HEATMAP_FILE = REPORTS_DIR / "correlation_heatmap.png"


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


def load_numeric_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dataset = ds.dataset(ANALYTICAL_DIR, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=SELECTED_COLUMNS)
    df = table.to_pandas()

    for column in NUMERIC_STRING_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    metadata_rows = []
    for column in df.columns:
        metadata_rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "null_count": int(df[column].isna().sum()),
                "null_pct": round(float(df[column].isna().mean()) * 100, 2),
                "nunique": int(df[column].nunique(dropna=True)),
                "std": round(float(df[column].std()), 6) if df[column].notna().any() else 0.0,
                "included_in_matrix": column not in EXCLUDED_COLUMNS,
                "exclusion_reason": EXCLUDED_COLUMNS.get(column, ""),
                "feature_status": (
                    "leakage_or_post_event"
                    if column in LEAKAGE_COLUMNS
                    else "candidate"
                ),
                "feature_family": FEATURE_FAMILY.get(column, "other"),
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    included_columns = [
        column for column in df.columns if column not in EXCLUDED_COLUMNS
    ]
    corr_df = df[included_columns].astype("float32")

    memory_mb = corr_df.memory_usage(deep=True).sum() / (1024 ** 2)
    load_stats = {
        "rows": len(corr_df),
        "columns": len(corr_df.columns),
        "memory_mb": round(memory_mb, 2),
        "selected_columns": list(corr_df.columns),
    }

    log.info(
        "Frame de correlacao carregado | rows=%s | cols=%s | memory_mb=%s",
        load_stats["rows"],
        load_stats["columns"],
        load_stats["memory_mb"],
    )
    return corr_df, metadata_df, load_stats


# ============================================================
# CALCULOS
# ============================================================

def build_target_correlations(corr_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    correlation_series = corr_df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)
    target_df = (
        correlation_series.rename("corr_to_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    target_df["abs_corr_to_target"] = target_df["corr_to_target"].abs()
    target_df = target_df.merge(
        metadata_df[["feature", "null_pct", "feature_status"]],
        on="feature",
        how="left",
    )
    target_df = target_df.merge(
        metadata_df[["feature", "feature_family"]],
        on="feature",
        how="left",
    )
    target_df["modeling_role"] = target_df["feature"].map(
        lambda value: "exclude"
        if value in LEAKAGE_COLUMNS
        else "candidate"
    )
    target_df["notes"] = target_df["feature"].map(
        lambda value: LEAKAGE_COLUMNS.get(value, "")
    )
    target_df = target_df.sort_values(
        by="abs_corr_to_target",
        ascending=False,
        ignore_index=True,
    )
    return target_df


def build_multicollinearity_pairs(corr_df: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = corr_df.corr(numeric_only=True)
    features = [column for column in corr_matrix.columns if column != TARGET_COLUMN]
    rows = []

    for index, left_feature in enumerate(features):
        for right_feature in features[index + 1:]:
            corr_value = corr_matrix.loc[left_feature, right_feature]
            if pd.notna(corr_value) and abs(corr_value) >= HIGH_CORR_THRESHOLD:
                rows.append(
                    {
                        "feature_left": left_feature,
                        "feature_right": right_feature,
                        "corr": round(float(corr_value), 6),
                        "abs_corr": round(abs(float(corr_value)), 6),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["feature_left", "feature_right", "corr", "abs_corr"]
        )

    return pd.DataFrame(rows).sort_values(
        by="abs_corr",
        ascending=False,
        ignore_index=True,
    )


def build_low_signal_candidates(target_df: pd.DataFrame) -> pd.DataFrame:
    candidates_df = target_df[
        (target_df["modeling_role"] == "candidate")
        & (target_df["abs_corr_to_target"] < LOW_TARGET_CORR_THRESHOLD)
    ].copy()
    candidates_df["possible_action"] = candidates_df["feature_family"].map(
        {
            "categorical_code": "keep_as_categorical_test_not_by_pearson_alone",
            "spatial_coordinate": "transform_into_distance_or_zone_features",
            "count": "keep_for_baseline_due_business_plausibility",
            "continuous": "keep_for_baseline_due_business_plausibility",
            "monetary_component": "deprioritize_raw_feature_or_validate_semantics",
        }
    ).fillna("deprioritize_or_exclude_after_model_baseline")
    return candidates_df


def build_correlation_matrix(corr_df: pd.DataFrame) -> pd.DataFrame:
    return corr_df.corr(numeric_only=True).round(6)


# ============================================================
# VISUALIZACAO
# ============================================================

def save_heatmap(correlation_matrix: pd.DataFrame) -> None:
    sns.set_theme(style="white")
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de correlacao das variaveis numericas")
    plt.tight_layout()
    plt.savefig(HEATMAP_FILE, dpi=160, bbox_inches="tight")
    plt.close()


# ============================================================
# RELATORIO
# ============================================================

def render_priority_lines(
    target_df: pd.DataFrame,
    low_signal_df: pd.DataFrame,
    multicollinear_df: pd.DataFrame,
) -> list[str]:
    lines = []

    eligible_df = target_df[target_df["modeling_role"] == "candidate"].copy()
    top_candidates = eligible_df.head(5)

    if not top_candidates.empty:
        top_text = "; ".join(
            f"{row.feature} ({row.corr_to_target:.4f})"
            for row in top_candidates.itertuples()
        )
        lines.append(
            "Sinal linear mais alto entre variaveis elegiveis: " + top_text + "."
        )

    if "RidePrice" in target_df["feature"].values:
        ride_price_corr = target_df.loc[
            target_df["feature"] == "RidePrice",
            "corr_to_target",
        ].iloc[0]
        lines.append(
            f"RidePrice tem correlacao {ride_price_corr:.4f} com o target, "
            "mas deve ser excluida por vazamento."
        )

    if not multicollinear_df.empty:
        pair_text = "; ".join(
            f"{row.feature_left} x {row.feature_right} ({row.corr:.4f})"
            for row in multicollinear_df.itertuples()
        )
        lines.append(
            "Multicolinearidade relevante detectada principalmente nas coordenadas de origem/destino: "
            + pair_text
            + "."
        )

    if not low_signal_df.empty:
        low_signal_text = ", ".join(low_signal_df["feature"].tolist())
        lines.append(
            "Baixo sinal linear global com o target nas colunas cruas: "
            + low_signal_text
            + ". Isso nao implica exclusao automatica para codigos categoricos ou variaveis espaciais transformaveis."
        )

    lines.append(
        "Para feature engineering, a prioridade deve ser transformar variaveis espaciais em distancia/zona, "
        "tratar ProductProviderID e CategoryID como categoricas e derivar features temporais a partir de Create/Schedule."
    )
    lines.append(
        "Selected e RideReasonSelectedEstimativeID nao devem entrar no treino porque sao campos posteriores ao evento de selecao."
    )

    return lines


def build_markdown_report(
    load_stats: dict,
    metadata_df: pd.DataFrame,
    target_df: pd.DataFrame,
    multicollinear_df: pd.DataFrame,
    low_signal_df: pd.DataFrame,
) -> str:
    included_df = metadata_df[metadata_df["included_in_matrix"]].copy()
    excluded_df = metadata_df[~metadata_df["included_in_matrix"]].copy()

    lines = [
        "# Analise de Correlacao com o Target Price",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{ANALYTICAL_DIR}`",
        f"- Carregamento em memoria: `{load_stats['rows']}` linhas x `{load_stats['columns']}` colunas (`~{load_stats['memory_mb']} MB` em `float32`).",
        "- Leitura feita diretamente do Parquet consolidado, selecionando apenas colunas numericas ou numerico-textuais convertiveis.",
        "",
        "## Variaveis Incluidas",
        "",
        included_df[
            ["feature", "dtype", "null_pct", "nunique", "feature_family", "feature_status"]
        ].to_markdown(index=False),
        "",
        "## Variaveis Excluidas da Matriz",
        "",
        excluded_df[
            ["feature", "null_pct", "nunique", "exclusion_reason"]
        ].to_markdown(index=False),
        "",
        "## Correlacao com o Target",
        "",
        target_df[
            ["feature", "corr_to_target", "abs_corr_to_target", "null_pct", "feature_family", "modeling_role", "notes"]
        ].to_markdown(index=False),
        "",
        "## Multicolinearidade",
        "",
        (
            multicollinear_df.to_markdown(index=False)
            if not multicollinear_df.empty
            else "Nenhum par acima do limiar configurado."
        ),
        "",
        "## Baixo Sinal Linear",
        "",
        (
            low_signal_df[
                ["feature", "corr_to_target", "abs_corr_to_target", "null_pct", "feature_family", "possible_action"]
            ].to_markdown(index=False)
            if not low_signal_df.empty
            else "Nenhuma variavel elegivel abaixo do limiar configurado."
        ),
        "",
        "## Interpretacao DS",
        "",
    ]

    for item in render_priority_lines(target_df, low_signal_df, multicollinear_df):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Conclusao",
            "",
            "- O unico sinal linear forte encontrado foi RidePrice, que precisa ser excluido por vazamento.",
            "- Entre variaveis elegiveis, a correlacao linear bruta com o target e baixa; isso sugere que o ganho virá mais de features derivadas e interacoes do que das colunas numericas cruas.",
            "- Coordenadas de origem/destino devem ser condensadas em features espaciais para reduzir redundancia e melhorar interpretabilidade.",
            f"- Heatmap salvo em `{HEATMAP_FILE.name}`.",
        ]
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_analytical_dataset_exists()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    corr_df, metadata_df, load_stats = load_numeric_frame()
    correlation_matrix = build_correlation_matrix(corr_df)
    target_df = build_target_correlations(corr_df, metadata_df)
    multicollinear_df = build_multicollinearity_pairs(corr_df)
    low_signal_df = build_low_signal_candidates(target_df)

    correlation_matrix.to_csv(CORRELATION_MATRIX_FILE)
    target_df.to_csv(TARGET_CORRELATIONS_FILE, index=False)
    multicollinear_df.to_csv(MULTICOLLINEARITY_FILE, index=False)
    low_signal_df.to_csv(LOW_SIGNAL_FILE, index=False)
    save_heatmap(correlation_matrix)
    REPORT_FILE.write_text(
        build_markdown_report(
            load_stats=load_stats,
            metadata_df=metadata_df,
            target_df=target_df,
            multicollinear_df=multicollinear_df,
            low_signal_df=low_signal_df,
        ),
        encoding="utf-8",
    )

    log.info("Matriz de correlacao salva em %s", CORRELATION_MATRIX_FILE)
    log.info("Correlacoes com o target salvas em %s", TARGET_CORRELATIONS_FILE)
    log.info("Multicolinearidade salva em %s", MULTICOLLINEARITY_FILE)
    log.info("Candidatas de baixo sinal salvas em %s", LOW_SIGNAL_FILE)
    log.info("Heatmap salvo em %s", HEATMAP_FILE)
    log.info("Relatorio salvo em %s", REPORT_FILE)


if __name__ == "__main__":
    main()
