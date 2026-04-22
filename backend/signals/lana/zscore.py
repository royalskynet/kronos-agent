import numpy as np
import pandas as pd


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window=window, min_periods=max(5, window // 3)).mean()
    std = s.rolling(window=window, min_periods=max(5, window // 3)).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)


def rolling_log_zscore(s: pd.Series, window: int) -> pd.Series:
    return rolling_zscore(np.log1p(s.clip(lower=0)), window)
