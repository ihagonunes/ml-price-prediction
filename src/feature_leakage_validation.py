from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from train import (
    TemporalSplitConfig,
    get_holdout_split,
    iter_tscv_splits,
    load_temporal_frame,
)


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

FEATURE_SOURCE_DIR = DATA_DIR / "features_temporal"
FOLD_CHECKS_FILE = REPORTS_DIR / "feature_leakage_fold_checks.csv"
RISK_TABLE_FILE = REPORTS_DIR / "feature_leakage_feature_risks.csv"
SUMMARY_METRICS_FILE = REPORTS_DIR / "feature_leakage_metrics.csv"
REPORT_FILE = REPORTS_DIR / "feature_leakage_validation.md"

TARGET_CATEGORY_NAMES = {
    2: "UberX",
    9: "Uber Comfort",
    4: "Uber Black",
}
OWN_FEATURE_BY_PRODUCT = {
    "UberX": "Price_UberX",
    "Comfort": "Price_Comfort",
    "Black": "Price_Black",
}
OTHER_FEATURES_BY_PRODUCT = {
    "UberX": ["Price_Comfort", "Price_Black"],
    "Comfort": ["Price_UberX", "Price_Black"],
    "Black": ["Price_UberX", "Price_Comfort"],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_feature_frame() -> pd.DataFrame:
    if not FEATURE_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset de features nao encontrado em {FEATURE_SOURCE_DIR}")

    dataset = ds.dataset(
        FEATURE_SOURCE_DIR,
        format="parquet",
        partitioning="hive",
    )
    columns = [
        "RideEstimativeID",
        "RideID",
        "UserID",
        "ProductID",
        "CategoryID",
        "Create",
        "Updated",
        "Price",
        "Price_UberX",
        "Price_Comfort",
        "Price_Black",
        "PriceWasCapped",
        "UserPriorRideCount",
        "UserPriorPaidPriceMean",
        "UserPriorCategoryRideCount",
        "UserPriorCategoryPriceMean",
    ]
    frame = dataset.to_table(columns=columns).to_pandas()
    frame["Create"] = pd.to_datetime(frame["Create"], errors="coerce")
    frame["Updated"] = pd.to_datetime(frame["Updated"], errors="coerce")
    frame["CategoryID"] = pd.to_numeric(frame["CategoryID"], errors="coerce").astype("Int64")
    if frame["Create"].isna().any():
        raise ValueError("Foram encontrados timestamps invalidos em Create no dataset de features.")

    frame = frame.sort_values(
        ["Create", "RideID", "RideEstimativeID"],
        kind="stable",
    ).reset_index(drop=True)
    return frame


def build_fold_check_rows(
    frame: pd.DataFrame,
    config: TemporalSplitConfig,
) -> pd.DataFrame:
    def summarize_split(
        scope: str,
        split_name: str,
        split_type: str,
        category_id: int | None,
        train_indices: np.ndarray,
        evaluation_indices: np.ndarray,
        gap_start: pd.Timestamp | None,
        gap_end: pd.Timestamp | None,
    ) -> dict:
        train_frame = frame.loc[train_indices]
        evaluation_frame = frame.loc[evaluation_indices]

        train_min_date = pd.Timestamp(train_frame["CreateDate"].min())
        train_max_date = pd.Timestamp(train_frame["CreateDate"].max())
        evaluation_min_date = pd.Timestamp(evaluation_frame["CreateDate"].min())
        evaluation_max_date = pd.Timestamp(evaluation_frame["CreateDate"].max())
        train_max_timestamp = pd.Timestamp(train_frame[config.time_column].max())
        evaluation_min_timestamp = pd.Timestamp(evaluation_frame[config.time_column].min())

        index_overlap = int(
            len(pd.Index(train_indices).intersection(pd.Index(evaluation_indices)))
        )
        train_rides = pd.Index(train_frame[config.ride_column].astype("int64"))
        evaluation_rides = pd.Index(evaluation_frame[config.ride_column].astype("int64"))
        ride_overlap = int(len(train_rides.intersection(evaluation_rides)))

        temporal_order_ok = bool(train_max_date < evaluation_min_date)
        timestamp_order_ok = bool(train_max_timestamp < evaluation_min_timestamp)
        gap_order_ok = True
        if gap_start is not None and gap_end is not None:
            gap_order_ok = bool((gap_start > train_max_date) and (evaluation_min_date > gap_end))

        fold_valid = bool(
            temporal_order_ok
            and timestamp_order_ok
            and gap_order_ok
            and index_overlap == 0
            and ride_overlap == 0
        )

        return {
            "scope": scope,
            "split_name": split_name,
            "split_type": split_type,
            "category_id": category_id,
            "category_name": TARGET_CATEGORY_NAMES.get(category_id, "all"),
            "train_rows": int(len(train_frame)),
            "evaluation_rows": int(len(evaluation_frame)),
            "train_start": train_min_date.date().isoformat(),
            "train_end": train_max_date.date().isoformat(),
            "gap_start": gap_start.date().isoformat() if gap_start is not None else "",
            "gap_end": gap_end.date().isoformat() if gap_end is not None else "",
            "evaluation_start": evaluation_min_date.date().isoformat(),
            "evaluation_end": evaluation_max_date.date().isoformat(),
            "index_overlap": index_overlap,
            "ride_overlap": ride_overlap,
            "temporal_order_ok": temporal_order_ok,
            "timestamp_order_ok": timestamp_order_ok,
            "gap_order_ok": gap_order_ok,
            "fold_valid": fold_valid,
        }

    rows = []
    for split_window, train_indices, evaluation_indices in iter_tscv_splits(frame, config):
        rows.append(
            summarize_split(
                scope="overall",
                split_name=split_window.name,
                split_type=split_window.split_type,
                category_id=None,
                train_indices=train_indices,
                evaluation_indices=evaluation_indices,
                gap_start=split_window.gap_start,
                gap_end=split_window.gap_end,
            )
        )

    holdout_window, train_indices, evaluation_indices = get_holdout_split(frame, config)
    rows.append(
        summarize_split(
            scope="overall",
            split_name=holdout_window.name,
            split_type=holdout_window.split_type,
            category_id=None,
            train_indices=train_indices,
            evaluation_indices=evaluation_indices,
            gap_start=holdout_window.gap_start,
            gap_end=holdout_window.gap_end,
        )
    )

    for category_id in config.target_categories:
        for split_window, train_indices, evaluation_indices in iter_tscv_splits(
            frame,
            config,
            category_id=category_id,
        ):
            rows.append(
                summarize_split(
                    scope="category",
                    split_name=split_window.name,
                    split_type=split_window.split_type,
                    category_id=category_id,
                    train_indices=train_indices,
                    evaluation_indices=evaluation_indices,
                    gap_start=split_window.gap_start,
                    gap_end=split_window.gap_end,
                )
            )

        holdout_window, train_indices, evaluation_indices = get_holdout_split(
            frame,
            config,
            category_id=category_id,
        )
        rows.append(
            summarize_split(
                scope="category",
                split_name=holdout_window.name,
                split_type=holdout_window.split_type,
                category_id=category_id,
                train_indices=train_indices,
                evaluation_indices=evaluation_indices,
                gap_start=holdout_window.gap_start,
                gap_end=holdout_window.gap_end,
            )
        )

    checks_df = pd.DataFrame(rows)
    checks_df["category_sort"] = checks_df["category_id"].fillna(-1).astype("int64")
    checks_df = checks_df.sort_values(
        ["scope", "category_sort", "split_type", "split_name"],
        kind="stable",
    ).drop(columns=["category_sort"])
    return checks_df.reset_index(drop=True)


def validate_user_history_features(feature_df: pd.DataFrame) -> dict:
    metrics: dict[str, int | float] = {}

    ride_level_consistency = (
        feature_df.groupby("RideID")["UserPriorRideCount"].nunique().gt(1).sum()
    )
    category_level_consistency = (
        feature_df.groupby(["RideID", "CategoryID"])["UserPriorCategoryRideCount"]
        .nunique()
        .gt(1)
        .sum()
    )
    metrics["ride_level_prior_count_inconsistencies"] = int(ride_level_consistency)
    metrics["category_level_prior_count_inconsistencies"] = int(category_level_consistency)

    ride_level_df = (
        feature_df.sort_values(["RideID", "Create", "RideEstimativeID"], kind="stable")
        .drop_duplicates(["RideID"], keep="first")
        .sort_values(["UserID", "Create", "RideID"], kind="stable")
        .reset_index(drop=True)
    )
    expected_prior_ride_count = (
        ride_level_df.groupby("UserID", sort=False).cumcount().astype("int32")
    )
    prior_ride_count_mismatch = int(
        (ride_level_df["UserPriorRideCount"].astype("int32") != expected_prior_ride_count).sum()
    )
    metrics["prior_ride_count_mismatch_rows"] = prior_ride_count_mismatch

    category_level_df = (
        feature_df.sort_values(
            ["RideID", "CategoryID", "Create", "RideEstimativeID"],
            kind="stable",
        )
        .drop_duplicates(["RideID", "CategoryID"], keep="first")
        .sort_values(
            ["UserID", "CategoryID", "Create", "RideID", "RideEstimativeID"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    expected_prior_category_count = (
        category_level_df.groupby(["UserID", "CategoryID"], sort=False)
        .cumcount()
        .astype("int32")
    )
    prior_category_count_mismatch = int(
        (
            category_level_df["UserPriorCategoryRideCount"].astype("int32")
            != expected_prior_category_count
        ).sum()
    )
    metrics["prior_category_count_mismatch_rows"] = prior_category_count_mismatch

    metrics["negative_user_prior_ride_count_rows"] = int(
        (feature_df["UserPriorRideCount"] < 0).sum()
    )
    metrics["negative_user_prior_category_count_rows"] = int(
        (feature_df["UserPriorCategoryRideCount"] < 0).sum()
    )
    metrics["rows_with_prior_category_count_and_null_mean"] = int(
        (
            feature_df["UserPriorCategoryRideCount"].gt(0)
            & feature_df["UserPriorCategoryPriceMean"].isna()
        ).sum()
    )
    metrics["rows_with_prior_ride_count_and_null_paid_mean"] = int(
        (
            feature_df["UserPriorRideCount"].gt(0)
            & feature_df["UserPriorPaidPriceMean"].isna()
        ).sum()
    )

    return metrics


def build_feature_risk_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    updated_after_create_pct = round(
        (
            feature_df["Updated"].notna()
            & (feature_df["Updated"] > feature_df["Create"])
        ).mean()
        * 100,
        4,
    )
    price_was_capped_pct = round(
        feature_df["PriceWasCapped"].fillna(False).astype("bool").mean() * 100,
        4,
    )

    own_feature_rows = []
    for product_name, own_feature in OWN_FEATURE_BY_PRODUCT.items():
        product_df = feature_df.loc[feature_df["ProductID"] == product_name].copy()
        if product_df.empty:
            continue
        equal_to_target_pct = round(
            (product_df[own_feature] == product_df["Price"]).mean() * 100,
            4,
        )
        own_feature_rows.append(
            (
                f"Cross own price ({own_feature})",
                "high",
                "Excluir da matriz de features do proprio modelo.",
                (
                    f"Para linhas de {product_name}, {own_feature} coincide com o target Price "
                    f"em {equal_to_target_pct}% das linhas, caracterizando leakage direto."
                ),
            )
        )

    cross_aux_rows = []
    for product_name, other_features in OTHER_FEATURES_BY_PRODUCT.items():
        product_df = feature_df.loc[feature_df["ProductID"] == product_name].copy()
        if product_df.empty:
            continue
        both_available_pct = round(
            product_df[other_features].notna().all(axis=1).mean() * 100,
            4,
        )
        cross_aux_rows.append(
            (
                f"Cross auxiliar ({product_name})",
                "medium",
                "Manter com controle de disponibilidade e sem usar coluna propria.",
                (
                    f"As duas features auxiliares estao disponiveis em {both_available_pct}% "
                    f"das linhas de {product_name}."
                ),
            )
        )

    rows = [
        {
            "feature_group": "Create* temporais",
            "leakage_risk": "low",
            "recommended_decision": "Manter no baseline.",
            "rationale": "Derivadas do timestamp de criacao da corrida, disponivel no momento de inferencia.",
        },
        {
            "feature_group": "Schedule* temporais",
            "leakage_risk": "low",
            "recommended_decision": "Manter, desde que Schedule esteja disponivel na predicao.",
            "rationale": "Timestamp de agendamento e pre-evento; nao depende de resultado futuro da corrida.",
        },
        {
            "feature_group": "Updated* temporais",
            "leakage_risk": "high",
            "recommended_decision": "Excluir dos modelos de preco.",
            "rationale": (
                f"Updated e posterior a Create em {updated_after_create_pct}% das linhas, "
                "indicando forte risco de carregar informacao pos-evento."
            ),
        },
        {
            "feature_group": "UserPriorRideCount / UserPriorPaidPriceMean",
            "leakage_risk": "low",
            "recommended_decision": "Manter no baseline.",
            "rationale": (
                "Calculadas com janela expansiva por UserID no nivel de RideID, "
                "sempre excluindo a corrida atual."
            ),
        },
        {
            "feature_group": "UserPriorCategoryRideCount / UserPriorCategoryPriceMean",
            "leakage_risk": "low",
            "recommended_decision": "Manter no baseline.",
            "rationale": (
                "Calculadas com janela expansiva por UserID + CategoryID no nivel canonico RideID+CategoryID, "
                "sem uso de informacao futura."
            ),
        },
        {
            "feature_group": "PriceWasCapped",
            "leakage_risk": "high",
            "recommended_decision": "Excluir dos modelos de preco.",
            "rationale": (
                f"Indicador derivado do proprio target Price no preprocessamento "
                f"(ativo em {price_was_capped_pct}% das linhas)."
            ),
        },
        {
            "feature_group": "WaitingTimeWasCapped / FareIDWasImputed",
            "leakage_risk": "low",
            "recommended_decision": "Uso opcional; nao sao necessarias no baseline.",
            "rationale": "Flags de qualidade/transformacao de entrada; nao usam informacao futura do target.",
        },
    ]

    for feature_group, risk, decision, rationale in own_feature_rows + cross_aux_rows:
        rows.append(
            {
                "feature_group": feature_group,
                "leakage_risk": risk,
                "recommended_decision": decision,
                "rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


def write_report(
    fold_checks_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    metrics: dict,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_checks_df.to_csv(FOLD_CHECKS_FILE, index=False)
    risk_df.to_csv(RISK_TABLE_FILE, index=False)
    metrics_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    metrics_df.to_csv(SUMMARY_METRICS_FILE, index=False)

    invalid_folds = int((~fold_checks_df["fold_valid"]).sum())
    high_risk_df = risk_df.loc[risk_df["leakage_risk"] == "high"].copy()

    report_lines = [
        "# Validacao do Pipeline de Features (Leakage e Consistencia)",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte de features: `{FEATURE_SOURCE_DIR}`",
        f"- Fonte dos splits: `{TemporalSplitConfig().source_dir}`",
        "",
        "## Resultado DE",
        "",
        (
            f"- Folds avaliados: `{len(fold_checks_df)}` "
            f"(`{invalid_folds}` com falha)."
        ),
        (
            f"- Overlap total de RideID entre treino e validacao: "
            f"`{int(fold_checks_df['ride_overlap'].sum())}`."
        ),
        (
            f"- Overlap total de indices entre treino e validacao: "
            f"`{int(fold_checks_df['index_overlap'].sum())}`."
        ),
        (
            "- Regra temporal validada em todos os folds: "
            "`max(train_date) < min(validation_date)`."
        ),
        "",
        "## Resultado DS",
        "",
        "- Features historicas por usuario foram revisadas como sem leakage quando usadas com Create como ancora temporal.",
        "- Features com risco alto foram identificadas e marcadas para exclusao do baseline de modelagem.",
        "",
        "## Checagens de Consistencia das Features Historicas",
        "",
        f"- Inconsistencias de UserPriorRideCount por RideID: `{metrics['ride_level_prior_count_inconsistencies']}`.",
        f"- Inconsistencias de UserPriorCategoryRideCount por RideID+CategoryID: `{metrics['category_level_prior_count_inconsistencies']}`.",
        f"- Mismatch contra cumcount esperado (ride): `{metrics['prior_ride_count_mismatch_rows']}`.",
        f"- Mismatch contra cumcount esperado (categoria): `{metrics['prior_category_count_mismatch_rows']}`.",
        (
            f"- Linhas com UserPriorCategoryRideCount > 0 e media nula: "
            f"`{metrics['rows_with_prior_category_count_and_null_mean']}`."
        ),
        (
            f"- Linhas com UserPriorRideCount > 0 e UserPriorPaidPriceMean nula: "
            f"`{metrics['rows_with_prior_ride_count_and_null_paid_mean']}` "
            "(esperado quando nao existe RidePrice historico valido > 0)."
        ),
        "",
        "## Tabela de Risco de Leakage por Grupo de Feature",
        "",
        risk_df.to_markdown(index=False),
        "",
        "## Features de Alto Risco (Acao Obrigatoria)",
        "",
        high_risk_df.to_markdown(index=False) if not high_risk_df.empty else "- Nenhuma.",
        "",
        "## Evidencia dos Folds",
        "",
        fold_checks_df.to_markdown(index=False),
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> dict:
    log.info("=" * 60)
    log.info("Validando pipeline de features contra leakage")
    log.info("=" * 60)

    config = TemporalSplitConfig()
    split_frame = load_temporal_frame(config)
    feature_df = load_feature_frame()

    fold_checks_df = build_fold_check_rows(split_frame, config)
    if not fold_checks_df["fold_valid"].all():
        invalid_rows = fold_checks_df.loc[~fold_checks_df["fold_valid"]]
        raise ValueError(
            "Foram encontrados folds invalidos na checagem temporal:\n"
            f"{invalid_rows[['scope', 'split_name', 'category_id']].to_string(index=False)}"
        )

    metrics = validate_user_history_features(feature_df)
    hard_fail_metrics = [
        "ride_level_prior_count_inconsistencies",
        "category_level_prior_count_inconsistencies",
        "prior_ride_count_mismatch_rows",
        "prior_category_count_mismatch_rows",
        "negative_user_prior_ride_count_rows",
        "negative_user_prior_category_count_rows",
        "rows_with_prior_category_count_and_null_mean",
    ]
    for metric_name in hard_fail_metrics:
        if int(metrics.get(metric_name, 0)) != 0:
            raise ValueError(
                f"Falha de consistencia historica detectada em {metric_name}: "
                f"{metrics[metric_name]}"
            )

    risk_df = build_feature_risk_table(feature_df)
    write_report(
        fold_checks_df=fold_checks_df,
        risk_df=risk_df,
        metrics=metrics,
    )

    log.info("Relatorio salvo em %s", REPORT_FILE)
    log.info("=" * 60)
    log.info("Validacao de leakage concluida com sucesso")
    log.info("=" * 60)
    return metrics


if __name__ == "__main__":
    main()
