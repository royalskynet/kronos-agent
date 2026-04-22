"""Lana 核心 edge：OI 漲但價格滯後 → MM 建倉特徵。

Binance `/futures/data/openInterestHist` 最細 5m 粒度、30d 回溯。
DataProvider 需把 OI 時序對齊 freqtrade 的 dataframe（同 timeframe）。
"""
import pandas as pd

from .zscore import rolling_zscore


def oi_price_divergence(
    df: pd.DataFrame,
    oi_col: str = "open_interest",
    price_col: str = "close",
    window: int = 48 * 12,  # 48h on 5m bars
) -> pd.Series:
    """z(ΔOI) - z(ΔPrice).

    正值大 = OI 顯著上升但價格沒跟上 → 主動 MM 悄悄建倉。
    負值 = 價格領先、OI 沒跟 → 追熱行情 (忽略)。
    """
    if oi_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    oi_ret = df[oi_col].pct_change(fill_method=None).fillna(0)
    px_ret = df[price_col].pct_change(fill_method=None).fillna(0)

    z_oi = rolling_zscore(oi_ret, window)
    z_px = rolling_zscore(px_ret.abs(), window)

    return z_oi - z_px


def oi_build_up_flag(
    df: pd.DataFrame, divergence_col: str = "oi_div", threshold: float = 1.5
) -> pd.Series:
    """純布林，方便 entry trigger。"""
    return (df[divergence_col] > threshold).astype(int)
