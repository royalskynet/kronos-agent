"""Funding rate 異動：短倉擁擠 (funding 轉負) 但 OI 仍升 = 軋空候選。"""
import pandas as pd

from .zscore import rolling_zscore


def funding_anomaly(
    df: pd.DataFrame,
    funding_col: str = "funding_rate",
    window: int = 96,  # ~8h on 5m
) -> pd.Series:
    if funding_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return -rolling_zscore(df[funding_col], window)  # 負 funding 越深分越高


def short_squeeze_setup(
    df: pd.DataFrame,
    anomaly_col: str = "funding_anom",
    oi_div_col: str = "oi_div",
) -> pd.Series:
    return ((df[anomaly_col] > 1.0) & (df[oi_div_col] > 1.0)).astype(int)
