from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# CONFIGURACAO
# ============================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass(frozen=True)
class TemporalSplitConfig:
    source_dir: Path = DATA_DIR / "analytical_curated"
    time_column: str = "Create"
    ride_column: str = "RideID"
    category_column: str = "CategoryID"
    regime_start: str = "2021-11-01"
    tscv_splits: int = 4
    validation_window_days: int = 28
    gap_days: int = 7
    holdout_window_days: int = 28
    target_categories: tuple[int, ...] = (2, 9, 4)


@dataclass(frozen=True)
class SplitWindow:
    name: str
    split_type: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    gap_start: pd.Timestamp | None
    gap_end: pd.Timestamp | None
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


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

def load_temporal_frame(
    config: TemporalSplitConfig,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if not config.source_dir.exists():
        raise FileNotFoundError(
            f"Dataset curado nao encontrado em {config.source_dir}"
        )

    requested_columns = columns or [
        "RideEstimativeID",
        config.ride_column,
        config.category_column,
        config.time_column,
    ]

    dataset = ds.dataset(
        config.source_dir,
        format="parquet",
        partitioning="hive",
    )
    frame = dataset.to_table(columns=requested_columns).to_pandas()
    frame[config.time_column] = pd.to_datetime(
        frame[config.time_column],
        errors="coerce",
    )
    if frame[config.time_column].isna().any():
        raise ValueError(
            f"Foram encontrados timestamps invalidos em {config.time_column}."
        )

    frame[config.category_column] = pd.to_numeric(
        frame[config.category_column],
        errors="coerce",
    ).astype("Int64")
    frame["CreateDate"] = frame[config.time_column].dt.floor("D")
    frame = frame.loc[frame["CreateDate"] >= pd.Timestamp(config.regime_start)].copy()
    frame = frame.sort_values(
        by=["CreateDate", config.time_column, config.ride_column],
        kind="stable",
    ).reset_index(drop=True)

    log.info(
        "Base temporal carregada | linhas=%s | rides=%s | datas=%s",
        len(frame),
        frame[config.ride_column].nunique(),
        frame["CreateDate"].nunique(),
    )
    return frame


# ============================================================
# SPLIT TEMPORAL
# ============================================================

def build_time_series_splitter(config: TemporalSplitConfig) -> TimeSeriesSplit:
    return TimeSeriesSplit(
        n_splits=config.tscv_splits,
        test_size=config.validation_window_days,
        gap=config.gap_days,
    )


def build_split_windows(
    frame: pd.DataFrame,
    config: TemporalSplitConfig,
) -> tuple[list[SplitWindow], SplitWindow]:
    unique_dates = pd.Index(sorted(frame["CreateDate"].dropna().unique()))
    required_days = (
        config.holdout_window_days
        + config.gap_days
        + (config.tscv_splits * config.validation_window_days)
        + config.gap_days
        + 1
    )
    if len(unique_dates) < required_days:
        raise ValueError(
            "Cobertura temporal insuficiente para o desenho solicitado. "
            f"Datas disponiveis={len(unique_dates)} | Minimo requerido={required_days}"
        )

    train_pool_dates = unique_dates[: -(config.holdout_window_days + config.gap_days)]
    holdout_gap_dates = unique_dates[
        -(config.holdout_window_days + config.gap_days) : -config.holdout_window_days
    ]
    holdout_test_dates = unique_dates[-config.holdout_window_days :]

    tscv = build_time_series_splitter(config)
    cv_windows: list[SplitWindow] = []
    for fold_number, (train_idx, test_idx) in enumerate(
        tscv.split(train_pool_dates),
        start=1,
    ):
        train_dates = train_pool_dates[train_idx]
        test_dates = train_pool_dates[test_idx]
        gap_dates = train_pool_dates[train_idx[-1] + 1 : test_idx[0]]

        cv_windows.append(
            SplitWindow(
                name=f"fold_{fold_number}",
                split_type="cv",
                train_start=pd.Timestamp(train_dates.min()),
                train_end=pd.Timestamp(train_dates.max()),
                gap_start=pd.Timestamp(gap_dates.min()) if len(gap_dates) else None,
                gap_end=pd.Timestamp(gap_dates.max()) if len(gap_dates) else None,
                evaluation_start=pd.Timestamp(test_dates.min()),
                evaluation_end=pd.Timestamp(test_dates.max()),
            )
        )

    holdout_window = SplitWindow(
        name="holdout_test",
        split_type="holdout",
        train_start=pd.Timestamp(train_pool_dates.min()),
        train_end=pd.Timestamp(train_pool_dates.max()),
        gap_start=pd.Timestamp(holdout_gap_dates.min()),
        gap_end=pd.Timestamp(holdout_gap_dates.max()),
        evaluation_start=pd.Timestamp(holdout_test_dates.min()),
        evaluation_end=pd.Timestamp(holdout_test_dates.max()),
    )
    return cv_windows, holdout_window


def apply_split_window(
    frame: pd.DataFrame,
    split_window: SplitWindow,
    config: TemporalSplitConfig,
    category_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    train_mask = frame["CreateDate"].between(
        split_window.train_start,
        split_window.train_end,
    )
    evaluation_mask = frame["CreateDate"].between(
        split_window.evaluation_start,
        split_window.evaluation_end,
    )

    if category_id is not None:
        category_mask = frame[config.category_column].eq(category_id)
        train_mask &= category_mask
        evaluation_mask &= category_mask

    train_indices = frame.index[train_mask].to_numpy(dtype=np.int64)
    evaluation_indices = frame.index[evaluation_mask].to_numpy(dtype=np.int64)
    return train_indices, evaluation_indices


def iter_tscv_splits(
    frame: pd.DataFrame,
    config: TemporalSplitConfig,
    category_id: int | None = None,
):
    cv_windows, _ = build_split_windows(frame, config)
    for split_window in cv_windows:
        yield (
            split_window,
            *apply_split_window(
                frame=frame,
                split_window=split_window,
                config=config,
                category_id=category_id,
            ),
        )


def get_holdout_split(
    frame: pd.DataFrame,
    config: TemporalSplitConfig,
    category_id: int | None = None,
) -> tuple[SplitWindow, np.ndarray, np.ndarray]:
    _, holdout_window = build_split_windows(frame, config)
    train_indices, evaluation_indices = apply_split_window(
        frame=frame,
        split_window=holdout_window,
        config=config,
        category_id=category_id,
    )
    return holdout_window, train_indices, evaluation_indices


# ============================================================
# VALIDACAO + SUMARIOS
# ============================================================

def summarize_split_window(
    frame: pd.DataFrame,
    split_window: SplitWindow,
    config: TemporalSplitConfig,
    category_id: int | None = None,
) -> dict:
    train_indices, evaluation_indices = apply_split_window(
        frame=frame,
        split_window=split_window,
        config=config,
        category_id=category_id,
    )
    train_frame = frame.loc[train_indices]
    evaluation_frame = frame.loc[evaluation_indices]

    train_ride_ids = set(train_frame[config.ride_column].astype("int64").tolist())
    evaluation_ride_ids = set(
        evaluation_frame[config.ride_column].astype("int64").tolist()
    )
    ride_overlap = len(train_ride_ids.intersection(evaluation_ride_ids))

    return {
        "split_name": split_window.name,
        "split_type": split_window.split_type,
        "category_id": category_id,
        "train_start": split_window.train_start.date().isoformat(),
        "train_end": split_window.train_end.date().isoformat(),
        "gap_start": (
            split_window.gap_start.date().isoformat()
            if split_window.gap_start is not None
            else ""
        ),
        "gap_end": (
            split_window.gap_end.date().isoformat()
            if split_window.gap_end is not None
            else ""
        ),
        "evaluation_start": split_window.evaluation_start.date().isoformat(),
        "evaluation_end": split_window.evaluation_end.date().isoformat(),
        "train_days": int(train_frame["CreateDate"].nunique()),
        "gap_days": (
            int((split_window.gap_end - split_window.gap_start).days + 1)
            if split_window.gap_start is not None and split_window.gap_end is not None
            else 0
        ),
        "evaluation_days": int(evaluation_frame["CreateDate"].nunique()),
        "train_rows": int(len(train_frame)),
        "evaluation_rows": int(len(evaluation_frame)),
        "train_unique_rides": int(train_frame[config.ride_column].nunique()),
        "evaluation_unique_rides": int(
            evaluation_frame[config.ride_column].nunique()
        ),
        "ride_overlap": ride_overlap,
    }


def build_summary_tables(
    frame: pd.DataFrame,
    config: TemporalSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_windows, holdout_window = build_split_windows(frame, config)
    all_windows = cv_windows + [holdout_window]

    overall_rows = [
        summarize_split_window(frame, split_window, config)
        for split_window in all_windows
    ]
    overall_df = pd.DataFrame(overall_rows)

    category_rows = []
    for split_window in all_windows:
        for category_id in config.target_categories:
            category_rows.append(
                summarize_split_window(
                    frame=frame,
                    split_window=split_window,
                    config=config,
                    category_id=category_id,
                )
            )
    category_df = pd.DataFrame(category_rows)
    return overall_df, category_df


def write_strategy_report(
    config: TemporalSplitConfig,
    frame: pd.DataFrame,
    overall_df: pd.DataFrame,
    category_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_file = REPORTS_DIR / "tscv_fold_summary.csv"
    category_file = REPORTS_DIR / "tscv_target_category_summary.csv"
    strategy_file = REPORTS_DIR / "tscv_strategy.md"

    overall_df.to_csv(overall_file, index=False)
    category_df.to_csv(category_file, index=False)

    holdout_row = overall_df.loc[overall_df["split_type"] == "holdout"].iloc[0]
    holdout_categories = category_df.loc[category_df["split_type"] == "holdout"].copy()

    target_volume_lines = []
    for category_id in config.target_categories:
        category_holdout = holdout_categories.loc[
            holdout_categories["category_id"] == category_id
        ].iloc[0]
        target_volume_lines.append(
            (
                f"- Categoria `{category_id}` no holdout: "
                f"{int(category_holdout['evaluation_rows'])} linhas e "
                f"{int(category_holdout['evaluation_unique_rides'])} corridas unicas."
            )
        )

    report_lines = [
        "# Estrategia Temporal de Train/Test e TSCV",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte: `{config.source_dir}`",
        f"- Regime usado para modelagem: `{config.regime_start}` ate `{frame['CreateDate'].max().date().isoformat()}`",
        "",
        "## Decisao DS",
        "",
        (
            "- O desenho parte do regime mais recente (`2021-11-01` em diante), porque a analise temporal "
            "mostrou quebra forte de volume entre `ago-out/2021` e `nov/2021-jun/2022`."
        ),
        (
            f"- O holdout final usa os ultimos `{config.holdout_window_days}` dias "
            f"(`{holdout_row['evaluation_start']}` a `{holdout_row['evaluation_end']}`), "
            f"com embargo de `{config.gap_days}` dias para evitar leakage temporal entre treino e teste."
        ),
        (
            f"- O TSCV interno usa `{config.tscv_splits}` folds expansivos, cada um com "
            f"`{config.validation_window_days}` dias de validacao e `gap` de `{config.gap_days}` dias."
        ),
        (
            "- A ancora temporal e `Create`, nao `Updated`, e o split e aplicado no nivel de dia "
            "para respeitar semanas completas e manter todas as estimativas da mesma corrida no mesmo lado do corte."
        ),
        "",
        "## Resumo Geral dos Splits",
        "",
        overall_df.to_markdown(index=False),
        "",
        "## Volume das Categorias-Alvo",
        "",
        category_df.to_markdown(index=False),
        "",
        "## Validacao de Integridade",
        "",
        (
            f"- Overlap de `RideID` entre treino e validacao/teste: "
            f"`{int(overall_df['ride_overlap'].sum())}` em todos os splits."
        ),
        (
            f"- Cobertura disponivel no regime escolhido: `{frame['CreateDate'].nunique()}` dias, "
            f"`{len(frame)}` linhas e `{frame[config.ride_column].nunique()}` corridas unicas."
        ),
        *target_volume_lines,
        "",
        "## Reuso nos 3 Pipelines",
        "",
        (
            "- `iter_tscv_splits(...)` expone os folds internos e `get_holdout_split(...)` retorna o corte final, "
            "com suporte opcional a filtro por `CategoryID`."
        ),
        (
            "- Isso permite que os tres pipelines de modelagem usem exatamente as mesmas janelas temporais, "
            "mantendo comparabilidade entre experimentos."
        ),
    ]
    strategy_file.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Relatorios salvos em %s", REPORTS_DIR)


# ============================================================
# MAIN
# ============================================================

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=" * 60)
    log.info("Definindo estrategia temporal de treino")
    log.info("=" * 60)

    config = TemporalSplitConfig()
    frame = load_temporal_frame(config)
    overall_df, category_df = build_summary_tables(frame, config)

    if int(overall_df["ride_overlap"].sum()) != 0:
        raise ValueError("Foram encontrados RideIDs compartilhados entre treino e avaliacao.")

    write_strategy_report(
        config=config,
        frame=frame,
        overall_df=overall_df,
        category_df=category_df,
    )

    log.info("=" * 60)
    log.info("Estrutura temporal pronta para reutilizacao")
    log.info("=" * 60)
    return overall_df, category_df


if __name__ == "__main__":
    main()
