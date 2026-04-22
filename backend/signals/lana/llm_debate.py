"""LLM Bull-Bear 1-round debate gate (TradingAgents inspired, minimal glue).

對 entry 候選 pair 跑 1 回合 Bull vs Bear，返回 conviction 0-1。
走 OpenRouter free (user_data.utils.llm.chat)。
Redis cache `llm_debate:{pair}` TTL 900s 控成本。

典型用法:
    result = debate("ORDI/USDT:USDT", {"hotness": 0.65, "z_oi": 1.8, ...})
    if result.conviction < 0.4:
        skip_entry()

單 call ≈ 300 tokens × 15min cache × 15 pair ≈ 免費 tier 綽綽有餘。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from user_data.utils.llm import chat as llm_chat

log = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("LLM_DEBATE_TTL", "900"))


@dataclass
class DebateResult:
    conviction: float      # 0.0-1.0
    bull_case: str         # ≤80 字
    bear_case: str         # ≤80 字
    verdict: str           # enter / skip / wait
    cached: bool = False


SYSTEM_PROMPT = """你是加密貨幣交易辯論裁判。對給定 pair 與訊號 features，模擬 Bull 與 Bear 1 回合辯論後輸出 JSON。

硬規:
- 只輸出 JSON，不包 markdown code block，不加前言
- bull_case / bear_case 各 <= 80 字，繁體中文
- conviction: 0.0-1.0 浮點 (0=確定別進, 1=確定進)
- verdict: "enter" (conv>=0.6) / "wait" (0.4-0.6) / "skip" (<0.4)

考量:
- 熱度 (hotness_score) 高 = Bull 加分
- Smart money 買入數 >=2 = Bull 加分
- funding rate 極端 (>0.05%) = Bear 加分
- macro risk-off = Bear 重擊
- BTC trend down = Bear 加分"""


def _redis():
    try:
        import os as _os
        import redis as _redis
        return _redis.Redis(
            host=_os.getenv("REDIS_HOST", "redis"),
            port=int(_os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_timeout=1.0,
        )
    except Exception:
        return None


def debate(pair: str, features: dict, max_tokens: int = 400) -> DebateResult:
    r = _redis()
    cache_key = f"llm_debate:{pair.replace('/', '_').replace(':', '')}"
    if r is not None:
        try:
            raw = r.get(cache_key)
            if raw:
                data = json.loads(raw)
                return DebateResult(
                    conviction=float(data.get("conviction", 0.0)),
                    bull_case=data.get("bull_case", ""),
                    bear_case=data.get("bear_case", ""),
                    verdict=data.get("verdict", "skip"),
                    cached=True,
                )
        except Exception:
            pass

    user_content = json.dumps({
        "pair": pair,
        "features": {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                     for k, v in features.items() if v is not None},
    }, ensure_ascii=False)

    try:
        raw = llm_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            json_mode=True,
            temperature=0.2,
            timeout=30,
            max_tokens=max_tokens,
        )
        if not raw:
            return DebateResult(0.0, "", "LLM 無回應", "skip")
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"[llm_debate] {pair} failed: {e}")
        return DebateResult(0.0, "", f"error:{str(e)[:40]}", "skip")

    result = DebateResult(
        conviction=max(0.0, min(1.0, float(data.get("conviction", 0.0)))),
        bull_case=str(data.get("bull_case", ""))[:200],
        bear_case=str(data.get("bear_case", ""))[:200],
        verdict=str(data.get("verdict", "skip")),
    )
    if r is not None:
        try:
            r.setex(cache_key, CACHE_TTL, json.dumps({
                "conviction": result.conviction,
                "bull_case": result.bull_case,
                "bear_case": result.bear_case,
                "verdict": result.verdict,
            }))
        except Exception:
            pass
    return result


if __name__ == "__main__":
    import sys
    pair = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT:USDT"
    feats = {"hotness_score": 0.7, "smart_money_buys": 3, "funding_rate": 0.0002}
    r = debate(pair, feats)
    print(r)
