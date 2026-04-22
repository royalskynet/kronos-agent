"""量放大但振幅壓縮 = 建倉結構。"""
import pandas as pd

from .zscore import rolling_log_zscore, rolling_zscore


def volume_compression(
    df: pd.DataFrame,
    window: int = 288,  # 24h on 5m
) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(0.0, index=df.index)

    vol_z = rolling_log_zscore(df["volume"], window)
    atr_like = (df["high"] - df["low"]) / df["close"]
    atr_z = rolling_zscore(atr_like, window)

    # 量大 + 振幅小 → 正分
    return vol_z - atr_z
