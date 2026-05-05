from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def prepare_output_dir(output_dir: Path, allowed_parent: Path) -> None:
    output_dir = output_dir.resolve()
    allowed_parent = allowed_parent.resolve()

    if output_dir != allowed_parent and allowed_parent not in output_dir.parents:
        raise ValueError(
            f"Diretorio de saida fora do escopo esperado: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def cast_object_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = list(df.select_dtypes(include=["object"]).columns)
    if not object_columns:
        return df

    converted = df.copy()
    for column in object_columns:
        converted[column] = converted[column].astype("string")
    return converted


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    converted = df.copy()
    for column in columns:
        if column in converted.columns:
            converted[column] = pd.to_numeric(
                converted[column],
                errors="coerce",
            )
    return converted


def parse_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    converted = df.copy()
    for column in columns:
        if column in converted.columns:
            converted[column] = pd.to_datetime(
                converted[column],
                errors="coerce",
            )
    return converted


def drop_duplicate_keys(
    df: pd.DataFrame,
    key_column: str,
    seen_keys: set[int],
) -> tuple[pd.DataFrame, int]:
    duplicate_mask = df[key_column].isin(seen_keys)
    deduped = df.loc[~duplicate_mask].copy()
    seen_keys.update(deduped[key_column].astype("int64").tolist())
    return deduped, int(duplicate_mask.sum())


def cap_upper_by_group(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    upper_bounds: dict[int, float],
) -> tuple[pd.DataFrame, pd.Series]:
    capped = df.copy()
    limit_series = capped[group_column].map(upper_bounds)
    if limit_series.isna().any():
        missing_groups = sorted(
            capped.loc[limit_series.isna(), group_column]
            .dropna()
            .astype("int64")
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Nao ha threshold configurado para todos os grupos: {missing_groups}"
        )

    cap_mask = capped[value_column] > limit_series
    capped.loc[cap_mask, value_column] = limit_series[cap_mask]
    return capped, cap_mask
