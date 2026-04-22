"""廣場/X 情緒合成分。經由 Redis 從 data_providers 拉。

Redis 缺席時優雅降級為中性 0 值，策略仍可跑（P0 dry-run 不需情緒層）。
"""
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_r = None
_redis_disabled = False


def _redis():
    global _r, _redis_disabled
    if _redis_disabled:
        return None
    if _r is None:
        try:
            import redis  # lazy import, 缺 package 也能載入本模組

            _r = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            _r.ping()
        except Exception as e:
            logger.info("Redis unavailable, sentiment layer disabled: %s", e)
            _redis_disabled = True
            return None
    return _r


NEUTRAL = {
    "square_post_z": 0.0,
    "square_unique_authors": 0,
    "square_human_weighted": 0.0,
    "square_bullish_count": 0,
    "square_bearish_count": 0,
    "square_neutral_count": 0,
    "square_bullish_ratio": 0.0,
    "square_sentiment_net": 0.0,
    "x_mention_z": 0.0,
    "x_kol_boost": 0.0,
    # 廣場全域 ticker 提及熱度 (WolfyXBT 風格)
    "square_ticker_mention_count": 0,
    "square_ticker_mention_z": 0.0,
    # CoinGecko Trending rank → 15..0 分
    "coingecko_trending_score": 0.0,
    # onchain (BGW)
    "onchain_supported": False,
    "onchain_holder_growth_z": 0.0,
    "onchain_smart_money_count": 0,
    "onchain_profit_ratio": 0.0,
    # Bitget 原生 OI (取代 Binance OI)
    "bitget_oi": 0.0,
    "bitget_oi_z": 0.0,
    # WolfyXBT 多源熱度
    "bitget_gainer_rank": 0.0,         # 0-1, Bitget 24h 漲幅榜排名
    "coingecko_trending_score": 0.0,   # 0-1, CoinGecko trending 排名
}


def sentiment_features(pair: str) -> dict:
    r = _redis()
    if r is None:
        return dict(NEUTRAL)
    try:
        key = f"sentiment:{pair.replace('/', '_').replace(':', '')}"
        raw = r.get(key)
        # 併入 Bitget 原生 OI (by bitget_oi.py)
        oi_key = f"bitget_oi:{pair.replace('/', '_').replace(':', '')}"
        oi_raw = r.get(oi_key)
    except Exception:
        return dict(NEUTRAL)
    if not raw and not oi_raw:
        return dict(NEUTRAL)
    try:
        feats = json.loads(raw) if raw else dict(NEUTRAL)
        if oi_raw:
            try:
                oi_data = json.loads(oi_raw)
                feats["bitget_oi"] = oi_data.get("bitget_oi", 0.0)
                feats["bitget_oi_z"] = oi_data.get("bitget_oi_z", 0.0)
            except json.JSONDecodeError:
                pass
        return feats
    except json.JSONDecodeError:
        return dict(NEUTRAL)


def hot_narrative_score(features: dict) -> float:
    """融合社群 + 鏈上信號.

    權重:
      - 廣場  post z-score         0.20
      - 廣場  作者多樣性           0.05
      - 廣場  qwen sentiment_net   0.15  (-1..+1 淨看漲比, 質而非量)
      - 廣場  全域 mention z       0.15  (WolfyXBT 風格: 全廣場 $TICKER 頻次)
      - X     提及 z               0.15
      - X     KOL boost            0.05
      - 鏈上  holder z             0.15
      - 鏈上  smart money          0.05
      - 鏈上  profit_ratio         0.05
    """
    sm_score = min(features.get("onchain_smart_money_count", 0), 10) / 10.0
    cg_norm = features.get("coingecko_trending_score", 0.0) / 15.0  # 0..1
    return (
        0.18 * features.get("square_post_z", 0.0)
        + 0.05 * features.get("square_human_weighted", 0.0)
        + 0.13 * features.get("square_sentiment_net", 0.0)
        + 0.13 * features.get("square_ticker_mention_z", 0.0)
        + 0.12 * cg_norm
        + 0.13 * features.get("x_mention_z", 0.0)
        + 0.04 * features.get("x_kol_boost", 0.0)
        + 0.12 * features.get("onchain_holder_growth_z", 0.0)
        + 0.05 * sm_score
        + 0.05 * features.get("onchain_profit_ratio", 0.0)
    )


def apply_sentiment(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    feats = sentiment_features(pair)
    for k, v in feats.items():
        df[k] = v
    df["sentiment_score"] = hot_narrative_score(feats)
    return df
