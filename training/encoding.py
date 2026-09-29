import numpy as np
import pandas as pd

from dataset import CATEGORICAL_COLUMNS, LOG_COLUMNS, NUMERIC_COLUMNS, TRANSFORM


def top_categories(values: pd.Series, top_n: int) -> list[str]:
    return values.value_counts().index[:top_n].tolist() + ["other"]


def fit_categories(df: pd.DataFrame, top_n: int) -> dict[str, list[str]]:
    return {col: top_categories(df[col], top_n) for col in CATEGORICAL_COLUMNS}


def bucket(values: pd.Series, categories: list[str]) -> pd.Series:
    known = set(categories) - {"other"}
    return values.where(values.isin(known), "other")


def onehot(values: pd.Series, categories: list[str]) -> np.ndarray:
    code_by_value = {value: i for i, value in enumerate(categories)}
    codes = bucket(values, categories).map(code_by_value).to_numpy()
    out = np.zeros((len(codes), len(categories)), dtype=np.float32)
    out[np.arange(len(codes)), codes] = 1.0
    return out


def column_transform(col: str) -> str:
    if col not in LOG_COLUMNS:
        return "none"
    return "log1p" if col == "dst_port" else TRANSFORM


def transform_map() -> dict[str, str]:
    return {col: column_transform(col) for col in NUMERIC_COLUMNS}


def log_transform_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        kind = column_transform(col)
        if kind == "log1p":
            numeric[col] = np.log1p(numeric[col].clip(lower=0))
        elif kind == "sqrt":
            numeric[col] = np.sqrt(numeric[col].clip(lower=0))
    return numeric


def numeric_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformed = log_transform_numeric(df)
    mean = transformed.mean().to_numpy(dtype=np.float32)
    std = transformed.std().to_numpy(dtype=np.float32)
    std[std == 0] = 1.0
    return mean, std


def standardize_numeric(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = log_transform_numeric(df)
    return ((transformed.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)


def encode(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, categories: dict[str, list[str]]) -> np.ndarray:
    parts = [standardize_numeric(df, mean, std)]
    for col in CATEGORICAL_COLUMNS:
        parts.append(onehot(df[col], categories[col]))
    return np.concatenate(parts, axis=1)
