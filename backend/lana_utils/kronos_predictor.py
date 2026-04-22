"""Kronos 時序預測 wrapper (shiyu-coder/Kronos).

非 LLM — decoder-only Transformer for OHLCV 序列。CPU 可跑 (mini=4.1M, small=24.7M)。

用途:
  - Path B entry pre-filter: predict next 15 bar close, up_prob > 0.55 才進
  - Early exit: predict 1h horizon, expected_return < -1% 早退
  - Redis cache TTL 300s 避免 per-bar 重算

Shadow 模式: log 預測 Redis 不影響 entry/exit; 1-2 週後驗收再上 live。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import NamedTuple, Optional

import pandas as pd

_log = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
_TOKENIZER_NAME = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
_CACHE_TTL = int(os.getenv("KRONOS_CACHE_TTL", "300"))
_MAX_CONTEXT = 512


class KronosResult(NamedTuple):
    mean_close: float       # 預測期 close 平均
    last_close: float       # 最後一根預測 close
    up_prob: float          # 預測期 close > 當前 close 比例 (0~1)
    expected_return: float  # (mean_close - current_close) / current_close
    ok: bool                # 推論成功
    cached: bool = False    # 來自 Redis cache


class _LazyPredictor:
    """單例：第一次呼叫才載入 HF model (避免 strategy 啟動卡)."""

    _inst = None

    def __init__(self):
        self._predictor = None
        self._load_failed = False

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def _load(self):
        if self._predictor is not None or self._load_failed:
            return
        try:
            from kronos import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
            tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_NAME)
            model = Kronos.from_pretrained(_MODEL_NAME)
            self._predictor = KronosPredictor(model, tokenizer, max_context=_MAX_CONTEXT)
            _log.info(f"[kronos] loaded {_MODEL_NAME}")
        except ImportError:
            _log.warning("[kronos] package not installed; predictions disabled")
            self._load_failed = True
        except Exception as e:
            _log.warning(f"[kronos] model load failed: {e}; predictions disabled")
            self._load_failed = True

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int = 15,
        T: float = 1.0,
        top_p: float = 0.9,
    ) -> Optional[pd.DataFrame]:
        self._load()
        if self._predictor is None:
            return None
        try:
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if not all(c in cols for c in ["open", "high", "low", "close"]):
                return None
            x_df = df[cols].reset_index(drop=True)
            x_timestamp = pd.Series(df.index if hasattr(df.index, "to_series") else range(len(df)))
            # 簡化: 以最後 bar 推未來 pred_len (timestamp 虛擬遞增 5min)
            last_ts = pd.Timestamp.utcnow() if x_timestamp.empty else pd.Timestamp(x_timestamp.iloc[-1])
            y_timestamp = pd.Series(
                pd.date_range(start=last_ts, periods=pred_len + 1, freq="5min")[1:]
            )
            return self._predictor.predict(
                df=x_df,
                x_timestamp=pd.Series(pd.to_datetime(x_timestamp, errors="coerce")).fillna(last_ts),
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=1,
            )
        except Exception as e:
            _log.warning(f"[kronos] predict failed: {e}")
            return None


def _redis():
    try:
        import redis
        return redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )
    except Exception:
        return None


def predict(pair: str, lookback_df: pd.DataFrame, pred_len: int = 15) -> KronosResult:
    """Main entry — 帶 Redis cache。"""
    current_close = float(lookback_df["close"].iloc[-1]) if len(lookback_df) else 0.0
    if current_close <= 0 or len(lookback_df) < 100:
        return KronosResult(0.0, 0.0, 0.5, 0.0, ok=False)

    r = _redis()
    cache_key = f"kronos:{pair.replace('/', '_').replace(':', '')}:{pred_len}"
    if r is not None:
        try:
            cached = r.get(cache_key)
            if cached:
                data = json.loads(cached)
                return KronosResult(
                    mean_close=data["mean_close"],
                    last_close=data["last_close"],
                    up_prob=data["up_prob"],
                    expected_return=data["expected_return"],
                    ok=True,
                    cached=True,
                )
        except Exception:
            pass

    pred_df = _LazyPredictor.get().predict(lookback_df.tail(_MAX_CONTEXT), pred_len=pred_len)
    if pred_df is None or "close" not in pred_df.columns or pred_df.empty:
        return KronosResult(0.0, 0.0, 0.5, 0.0, ok=False)

    mean_close = float(pred_df["close"].mean())
    last_close = float(pred_df["close"].iloc[-1])
    up_prob = float((pred_df["close"] > current_close).mean())
    expected_return = (mean_close - current_close) / current_close

    result = KronosResult(mean_close, last_close, up_prob, expected_return, ok=True)
    if r is not None:
        try:
            r.setex(cache_key, _CACHE_TTL, json.dumps({
                "mean_close": mean_close,
                "last_close": last_close,
                "up_prob": up_prob,
                "expected_return": expected_return,
                "ts": int(time.time()),
            }))
        except Exception:
            pass
    return result


def log_shadow(pair: str, result: KronosResult, current_close: float) -> None:
    """Shadow 模式: 把 Kronos 訊號寫 Redis 供 shadow_pnl_report 比對。"""
    if not result.ok:
        return
    r = _redis()
    if r is None:
        return
    ts = int(time.time())
    payload = {
        "pair": pair,
        "entry_close": current_close,
        "up_prob": result.up_prob,
        "expected_return": result.expected_return,
        "fired": bool(result.up_prob > 0.55),
        "ts": ts,
    }
    try:
        r.setex(f"shadow_kronos:{pair}:{ts}", 86400, json.dumps(payload))
    except Exception:
        pass
