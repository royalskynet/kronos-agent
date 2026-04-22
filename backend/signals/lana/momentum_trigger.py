"""P1 Shadow Path B — momentum/熱度觸發訊號 (only logged, not used for entry).

對齊 Lana 原生「熱度先選 → 順勢進場」邏輯。
當前 strategy entry 走 OI 背離 (Path A)，此模組獨立計算 Path B 訊號，
僅 log 不影響開單，1-2 週後對比 hypothetical PnL 決定是否上線 (P2)。
"""
from __future__ import annotations

import pandas as pd

from user_data.signals.zscore import rolling_zscore


def momentum_path_b(
    df: pd.DataFrame,
    sentiment_feats: dict,
    pct_change_min: float = 3.0,
    vol_z_min: float = 0.5,
    mention_z_min: float = 0.3,
) -> pd.Series:
    """棍哥風格寬進 — 結構 AND + 熱度 OR (任一熱度源命中即可).

    Structure (AND):
      - 24h pct change > pct_change_min %  (預設 3%)
      - rolling 8h volume z-score > vol_z_min  (預設 0.5)
      - close > VWAP(48 bar = 4h)

    Hotness (OR, 任一熱度源有動靜即可):
      - square_mention_z > mention_z_min  OR
      - x_kol_boost > 0.5 (有 KOL 提到)  OR
      - coingecko_trending_score > 0.3 OR
      - bitget_gainer_rank > 0.3
    """
    if "close" not in df.columns or len(df) < 288:
        return pd.Series(False, index=df.index)

    pct_24h = df["close"].pct_change(288).fillna(0) * 100
    vol_z = rolling_zscore(df["volume"], 96).fillna(0)
    typical = (df["close"] * df["volume"]).rolling(48, min_periods=12).sum()
    vol_sum = df["volume"].rolling(48, min_periods=12).sum()
    vwap = (typical / vol_sum.where(vol_sum > 0, 1)).fillna(df["close"])

    # 熱度 OR: 任一源有動靜
    square_z = float(sentiment_feats.get("square_ticker_mention_z", 0) or 0)
    kol_boost = float(sentiment_feats.get("x_kol_boost", 0) or 0)
    cg_score = float(sentiment_feats.get("coingecko_trending_score", 0) or 0)
    bg_rank = float(sentiment_feats.get("bitget_gainer_rank", 0) or 0)
    hotness_hit = (
        square_z > mention_z_min
        or kol_boost > 0.5
        or cg_score > 0.3
        or bg_rank > 0.3
    )

    structure = (
        (pct_24h > pct_change_min)
        & (vol_z > vol_z_min)
        & (df["close"] > vwap)
    )
    # hotness_hit 是 scalar; broadcast 到 series
    return structure if hotness_hit else pd.Series(False, index=df.index)
