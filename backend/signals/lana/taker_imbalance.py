"""Taker 主動買賣失衡：/futures/data/takerlongshortRatio 的 z-score 突變。"""
import pandas as pd

from .zscore import rolling_zscore


def taker_imbalance(
    df: pd.DataFrame,
    ratio_col: str = "taker_ls_ratio",
    window: int = 96,
) -> pd.Series:
    if ratio_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return rolling_zscore(df[ratio_col], window)
