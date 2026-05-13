from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FINAL_FEATURES_DIR = DATA_DIR / "final_features"

BASELINE_FOLD_METRICS_FILE = REPORTS_DIR / "baseline_tscv_fold_metrics.csv"
BASELINE_SUMMARY_FILE = REPORTS_DIR / "baseline_tscv_summary.csv"
BASELINE_REPORT_FILE = REPORTS_DIR / "baseline_tscv_report.md"

CATEGORY_DATASETS = {
    2: {
        "dataset_name": "UberX",
        "file_name": "features_uberx.parquet",
        "auxiliary_cross_columns": ["Price_Comfort", "Price_Black"],
    },
    9: {
        "dataset_name": "Uber Comfort",
        "file_name": "features_comfort.parquet",
        "auxiliary_cross_columns": ["Price_UberX", "Price_Black"],
    },
    4: {
        "dataset_name": "Uber Black",
        "file_name": "features_black.parquet",
        "auxiliary_cross_columns": ["Price_UberX", "Price_Comfort"],
    },
}

NUMERIC_FEATURE_COLUMNS = [
    "WaitingTime",
    "Fee",
    "TotalUsers",
    "RideStatusID",
    "CompanyID",
    "ProductProviderID",
    "OriginLat",
    "OriginLng",
    "DestinationLat",
    "DestinationLng",
    "ScheduleHour",
    "ScheduleDayOfWeek",
    "ScheduleMonth",
    "ScheduleQuarter",
    "CreateHour",
    "CreateDayOfWeek",
    "CreateMonth",
    "CreateQuarter",
    "UserPriorRideCount",
    "UserPriorPaidPriceMean",
    "UserPriorCategoryRideCount",
    "UserPriorCategoryPriceMean",
]
BOOLEAN_FEATURE_COLUMNS = [
    "ScheduleIsHolidayBR",
    "CreateIsHolidayBR",
    "FareIDWasImputed",
    "WaitingTimeWasCapped",
]
CATEGORICAL_FEATURE_COLUMNS = [
    "ProductID",
]
TARGET_COLUMN = "Price"
TIME_ORDER_COLUMN = "CreateDate"


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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


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
    frame[TIME_ORDER_COLUMN] = frame[config.time_column].dt.floor("D")
    frame = frame.loc[frame[TIME_ORDER_COLUMN] >= pd.Timestamp(config.regime_start)].copy()
    frame = frame.sort_values(
        by=[TIME_ORDER_COLUMN, config.time_column, config.ride_column],
        kind="stable",
    ).reset_index(drop=True)

    log.info(
        "Base temporal carregada | linhas=%s | rides=%s | datas=%s",
        len(frame),
        frame[config.ride_column].nunique(),
        frame[TIME_ORDER_COLUMN].nunique(),
    )
    return frame


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
    unique_dates = pd.Index(sorted(frame[TIME_ORDER_COLUMN].dropna().unique()))
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
    train_mask = frame[TIME_ORDER_COLUMN].between(
        split_window.train_start,
        split_window.train_end,
    )
    evaluation_mask = frame[TIME_ORDER_COLUMN].between(
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
        "train_days": int(train_frame[TIME_ORDER_COLUMN].nunique()),
        "gap_days": (
            int((split_window.gap_end - split_window.gap_start).days + 1)
            if split_window.gap_start is not None and split_window.gap_end is not None
            else 0
        ),
        "evaluation_days": int(evaluation_frame[TIME_ORDER_COLUMN].nunique()),
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
        f"- Regime usado para modelagem: `{config.regime_start}` ate `{frame[TIME_ORDER_COLUMN].max().date().isoformat()}`",
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
            f"- Cobertura disponivel no regime escolhido: `{frame[TIME_ORDER_COLUMN].nunique()}` dias, "
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


def load_category_feature_frame(
    category_id: int,
    config: TemporalSplitConfig,
) -> pd.DataFrame:
    dataset_spec = CATEGORY_DATASETS[category_id]
    dataset_path = FINAL_FEATURES_DIR / dataset_spec["file_name"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset final nao encontrado em {dataset_path}")

    frame = ds.dataset(dataset_path, format="parquet").to_table().to_pandas()
    frame[config.time_column] = pd.to_datetime(
        frame[config.time_column],
        errors="coerce",
    )
    if frame[config.time_column].isna().any():
        raise ValueError(
            f"Foram encontrados timestamps invalidos em {config.time_column} para {dataset_spec['dataset_name']}."
        )

    frame[TIME_ORDER_COLUMN] = pd.to_datetime(
        frame[TIME_ORDER_COLUMN],
        errors="coerce",
    )
    if frame[TIME_ORDER_COLUMN].isna().any():
        raise ValueError(
            f"Foram encontrados timestamps invalidos em {TIME_ORDER_COLUMN} para {dataset_spec['dataset_name']}."
        )

    frame = frame.loc[frame[TIME_ORDER_COLUMN] >= pd.Timestamp(config.regime_start)].copy()
    frame = frame.sort_values(
        by=[TIME_ORDER_COLUMN, config.time_column, config.ride_column, "RideEstimativeID"],
        kind="stable",
    ).reset_index(drop=True)

    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("int8")
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")

    log.info(
        "Dataset final carregado | categoria=%s | linhas=%s | datas=%s",
        dataset_spec["dataset_name"],
        len(frame),
        frame[TIME_ORDER_COLUMN].nunique(),
    )
    return frame


def get_feature_columns(
    category_id: int,
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    dataset_spec = CATEGORY_DATASETS[category_id]
    numeric_columns = (
        NUMERIC_FEATURE_COLUMNS
        + BOOLEAN_FEATURE_COLUMNS
        + dataset_spec["auxiliary_cross_columns"]
    )

    missing_numeric = sorted(set(numeric_columns).difference(frame.columns))
    missing_categorical = sorted(set(CATEGORICAL_FEATURE_COLUMNS).difference(frame.columns))
    if missing_numeric or missing_categorical:
        raise ValueError(
            "Colunas esperadas ausentes na base final. "
            f"Numericas faltantes={missing_numeric} | Categoricas faltantes={missing_categorical}"
        )

    return numeric_columns, CATEGORICAL_FEATURE_COLUMNS


def build_baseline_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Pipeline]:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
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

    return {
        "LinearRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression(n_jobs=-1)),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    Lasso(
                        alpha=0.001,
                        max_iter=20000,
                        tol=1e-3,
                        selection="random",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def apply_split_window_to_category_frame(
    frame: pd.DataFrame,
    split_window: SplitWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = frame[TIME_ORDER_COLUMN].between(
        split_window.train_start,
        split_window.train_end,
    )
    evaluation_mask = frame[TIME_ORDER_COLUMN].between(
        split_window.evaluation_start,
        split_window.evaluation_end,
    )
    train_frame = frame.loc[train_mask].copy()
    evaluation_frame = frame.loc[evaluation_mask].copy()
    return train_frame, evaluation_frame


def compute_regression_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "MAPE": float(mean_absolute_percentage_error(y_true, predictions) * 100),
        "R2": float(r2_score(y_true, predictions)),
    }


def run_baseline_tscv(
    config: TemporalSplitConfig,
    cv_windows: list[SplitWindow],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict] = []

    for category_id in config.target_categories:
        dataset_spec = CATEGORY_DATASETS[category_id]
        category_frame = load_category_feature_frame(category_id, config)
        numeric_columns, categorical_columns = get_feature_columns(
            category_id,
            category_frame,
        )

        for split_window in cv_windows:
            train_frame, evaluation_frame = apply_split_window_to_category_frame(
                category_frame,
                split_window,
            )
            if train_frame.empty or evaluation_frame.empty:
                raise ValueError(
                    f"Split sem dados para {dataset_spec['dataset_name']} em {split_window.name}."
                )

            ride_overlap = len(
                set(train_frame[config.ride_column].astype("int64"))
                .intersection(set(evaluation_frame[config.ride_column].astype("int64")))
            )
            if ride_overlap != 0:
                raise ValueError(
                    f"RideIDs compartilhados entre treino e validacao para {dataset_spec['dataset_name']} em {split_window.name}."
                )

            x_train = train_frame[numeric_columns + categorical_columns].copy()
            y_train = train_frame[TARGET_COLUMN].astype("float64")
            x_evaluation = evaluation_frame[numeric_columns + categorical_columns].copy()
            y_evaluation = evaluation_frame[TARGET_COLUMN].astype("float64")

            model_registry = build_baseline_models(
                numeric_columns,
                categorical_columns,
            )
            for model_name, model_pipeline in model_registry.items():
                model_pipeline.fit(x_train, y_train)
                predictions = model_pipeline.predict(x_evaluation)
                metrics = compute_regression_metrics(y_evaluation, predictions)
                feature_count = len(
                    model_pipeline.named_steps["preprocessor"].get_feature_names_out()
                )

                fold_rows.append(
                    {
                        "CategoryID": category_id,
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
                    "Baseline treinado | categoria=%s | modelo=%s | fold=%s | MAE=%.4f | RMSE=%.4f | MAPE=%.2f | R2=%.4f",
                    dataset_spec["dataset_name"],
                    model_name,
                    split_window.name,
                    metrics["MAE"],
                    metrics["RMSE"],
                    metrics["MAPE"],
                    metrics["R2"],
                )

    fold_metrics_df = pd.DataFrame(fold_rows).sort_values(
        by=["CategoryID", "ModelName", "FoldName"],
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
        .sort_values(by=["CategoryID", "MeanMAE", "MeanRMSE"], kind="stable")
        .reset_index(drop=True)
    )
    return fold_metrics_df, summary_df


def write_baseline_report(
    config: TemporalSplitConfig,
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(BASELINE_FOLD_METRICS_FILE, index=False)
    summary_df.to_csv(BASELINE_SUMMARY_FILE, index=False)

    best_by_category = (
        summary_df.sort_values(by=["CategoryID", "MeanMAE", "MeanRMSE"], kind="stable")
        .groupby(["CategoryID", "DatasetName"], as_index=False)
        .first()
    )
    best_lines = [
        (
            f"- `{row.DatasetName}`: melhor baseline por MAE foi `{row.ModelName}` "
            f"(MAE medio `{row.MeanMAE:.4f}`, RMSE medio `{row.MeanRMSE:.4f}`, "
            f"MAPE medio `{row.MeanMAPE:.2f}%`, R2 medio `{row.MeanR2:.4f}`)."
        )
        for row in best_by_category.itertuples(index=False)
    ]

    report_lines = [
        "# Baseline Models com TSCV",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Fonte dos datasets finais: `{FINAL_FEATURES_DIR}`",
        f"- Regime modelado: `{config.regime_start}` em diante.",
        "",
        "## Resultado DE",
        "",
        "- O loop de TSCV foi executado para as tres categorias-alvo e para os tres modelos baseline (`LinearRegression`, `Ridge` e `Lasso`).",
        "- As metricas por fold e as medias gerais foram persistidas em CSV para comparacao com os algoritmos avancados.",
        "",
        "## Resultado DS",
        "",
        "- Os baselines abaixo definem o piso minimo que os modelos avancados precisam superar.",
        "- A leitura usa a camada final por categoria, com as colunas de leakage alto ja removidas.",
        "- Para manter os baselines lineares estaveis e rapidos, o treino usa um subconjunto model-ready: variaveis numericas, flags booleanas e `ProductID` com one-hot.",
        "",
        "## Media por Modelo e Categoria",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Melhor Baseline por Categoria",
        "",
        *best_lines,
        "",
        "## Detalhe por Fold",
        "",
        fold_metrics_df.to_markdown(index=False),
    ]
    BASELINE_REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Relatorios baseline salvos em %s", REPORTS_DIR)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("=" * 60)
    log.info("Treinando modelos baseline com TSCV")
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

    cv_windows, _ = build_split_windows(frame, config)
    fold_metrics_df, summary_df = run_baseline_tscv(config, cv_windows)
    write_baseline_report(
        config=config,
        fold_metrics_df=fold_metrics_df,
        summary_df=summary_df,
    )

    log.info("=" * 60)
    log.info("Baselines prontos para comparacao com modelos avancados")
    log.info("=" * 60)
    return overall_df, category_df, fold_metrics_df, summary_df


if __name__ == "__main__":
    main()
