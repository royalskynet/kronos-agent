from __future__ import annotations

import math
from typing import Any


def _zscore(values: list[float], window: int) -> float | None:
    if len(values) < max(4, window // 4):
        return None
    sample = values[-window:]
    n = len(sample)
    mean = sum(sample) / n
    variance = sum((x - mean) ** 2 for x in sample) / n
    std = math.sqrt(variance) if variance > 0 else 0
    if std == 0:
        return 0.0
    return (sample[-1] - mean) / std


def _volume_spike(klines: list[dict[str, Any]], window: int = 20) -> dict[str, Any]:
    vols = [float(k.get("volume") or 0) for k in klines if k.get("volume") is not None]
    if len(vols) < 4:
        return {"score": 0.0, "detail": "insufficient data"}
    z = _zscore(vols, min(window, len(vols)))
    return {
        "score": round(z or 0.0, 3),
        "detail": f"vol_z={z:.2f}" if z is not None else "n/a",
        "signal": "HIGH" if (z or 0) > 1.5 else "LOW" if (z or 0) < -1.0 else "NORMAL",
    }


def _price_momentum(klines: list[dict[str, Any]], short: int = 5, long: int = 20) -> dict[str, Any]:
    closes = [float(k.get("close") or 0) for k in klines if k.get("close")]
    if len(closes) < long + 1:
        return {"score": 0.0, "detail": "insufficient data", "signal": "NEUTRAL"}
    short_avg = sum(closes[-short:]) / short
    long_avg = sum(closes[-long:]) / long
    pct = (short_avg - long_avg) / long_avg * 100 if long_avg else 0
    signal = "BULLISH" if pct > 0.5 else "BEARISH" if pct < -0.5 else "NEUTRAL"
    return {
        "score": round(pct, 3),
        "detail": f"short_ma/long_ma pct={pct:.2f}%",
        "signal": signal,
    }


def _funding_signal(funding_pct: float | None) -> dict[str, Any]:
    if funding_pct is None:
        return {"score": 0.0, "detail": "no data", "signal": "NEUTRAL"}
    pct = float(funding_pct)
    # negative funding = shorts crowded = potential squeeze
    signal = "SQUEEZE_SETUP" if pct < -0.01 else "LONGS_CROWDED" if pct > 0.05 else "NEUTRAL"
    return {
        "score": round(-pct * 100, 3),
        "detail": f"funding={pct:.4f}%",
        "signal": signal,
    }


def compute_lana_signals(candidate: dict[str, Any]) -> dict[str, Any]:
    klines_by_interval: dict[str, list] = candidate.get("klinesByInterval") or {}
    klines = (
        klines_by_interval.get("15m")
        or klines_by_interval.get("5m")
        or klines_by_interval.get("1m")
        or []
    )
    funding_pct = candidate.get("fundingPct")

    vol = _volume_spike(klines)
    mom = _price_momentum(klines)
    fund = _funding_signal(funding_pct)

    # composite: positive = opportunity, negative = caution
    composite = round(vol["score"] * 0.3 + mom["score"] * 0.4 + fund["score"] * 0.3, 3)

    return {
        "volume_spike": vol,
        "price_momentum": mom,
        "funding_signal": fund,
        "composite_score": composite,
        "note": "lana_signals v1: volume+momentum+funding. OI/taker require exchange subscription.",
    }
