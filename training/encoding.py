import numpy as np
import pandas as pd

from dataset import NUMERIC_COLUMNS


def top_proto_categories(proto: pd.Series, top_n: int) -> list[str]:
    counts = proto.value_counts()
    top = counts.index[:top_n].tolist()
    return top + ["other"]


def bucket_proto(proto: pd.Series, categories: list[str]) -> pd.Series:
    known = set(categories) - {"other"}
    return proto.where(proto.isin(known), "other")


def onehot_proto(proto: pd.Series, categories: list[str]) -> np.ndarray:
    code_by_value = {value: i for i, value in enumerate(categories)}
    codes = bucket_proto(proto, categories).map(code_by_value).to_numpy()
    onehot = np.zeros((len(codes), len(categories)), dtype=np.float32)
    onehot[np.arange(len(codes)), codes] = 1.0
    return onehot


def log_transform_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_COLUMNS].copy()
    numeric["dst_port"] = np.log1p(numeric["dst_port"])
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
