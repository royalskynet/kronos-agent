"""LanaDecisionAgent — 進出場前 sanity check, 用 OpenRouter 免費模型推理.

思路:
- freqtrade 規則層 (entry_score > 1.8, oi_div > 1.0) 是「機械訊號」
- 訊號觸發後, 呼叫 LLM 做「脈絡判讀」:
  * sentiment-analyst 邏輯: crowd positioning, overleveraged?
  * technical-analysis 邏輯: 5m 是否過熱? 15m 趨勢一致?
  * market-intel / news-briefing 留 P1+ 再加
- 回傳 (allow: bool, reason: str, confidence: float)
- LLM 掛掉 → fallback to allow (不要阻塞訊號)
- 決策全部寫 log 供覆盤

成本: OpenRouter nemotron/qwen/llama :free 輪替, 免費. 每次 entry 一次 call, ~1-3s.
P2+ 可升級 Haiku 4.5 提升推理.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import httpx

from user_data.utils.llm import chat as llm_chat

logger = logging.getLogger(__name__)

DECISION_TIMEOUT = float(os.getenv("DECISION_TIMEOUT", "12"))

TG_BOT = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT = os.getenv("TG_CHAT_ID", "")


def tg_send(text: str):
    if not TG_BOT or not TG_CHAT:
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{TG_BOT}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text},
            timeout=5,
        )
    except Exception as e:
        logger.debug("tg_send fail: %s", e)


EXIT_SYSTEM_PROMPT = """你是 Lana 出場 sanity-check, time_stop (持倉 N 小時無進展) 觸發時問你要不要延後平倉.

規則層會平, 但你看脈絡有沒有反轉跡象. 判讀維度:
1. OI 背離是否仍然正向? (+1 以上 = 建倉訊號還在, 可能要拉動)
2. taker_z 是否轉強? (> +1 = 主動買方進來)
3. funding 是否不極端? (|f| < 0.05% 安全)
4. 散戶偏移是否鬆動? (skew 下降 = 多頭擁擠緩解)
5. 價格是否在 BB lower 站穩? (recent_ret 仍微正, 未破整數關)

回 JSON: {"hold": true|false, "reason": "<20 字內>", "confidence": 0.0-1.0}
hold=true 且 confidence > 0.7 → 延後 4h 再看
其他情況 → 平倉

原則: 原則是「快出不磨盤」(Lana 哲學), 只有強信號還在才延後. 模糊投 hold=false."""


ENTRY_SYSTEM_PROMPT = """你是 Lana 交易 agent 的進場守門員. 規則層 (entry_score>1.8 + oi_div>1.0) 已高度確認訊號,
你的任務是「否決明顯陷阱」而不是「挑毛病」.

**預設放行**, 只在下列 **明確反訊號** 出現時 veto:

拒絕條件 (任一命中才拒):
A. 散戶嚴重擁擠 → net_position_skew > 0.5 (> +0.5 拒; 0 或小值當作沒數據, 放行)
B. Funding 過熱 → |funding_rate| > 0.0015 (> 0.15%/8h)
C. 主動買賣嚴重相反 → taker_z < -1.5 (明顯賣方壓倒)
D. 短期大漲後追高 → recent_ret_1h > +5% (已經被炒起來了, 追高陷阱)
E. OI 背離反向 → oi_div < 0.5 (規則層漏了本不該觸發)

其他情況 (包括數據為 0 或 "方向不明") → **allow: true**.
若無拒絕理由, 就給 allow: true, reason 寫「信號確認, 放行」, confidence 0.7+.

回 JSON: {"allow": true|false, "reason": "<20 字內中文>", "confidence": 0.0-1.0}

重要: 規則層已嚴格過濾. 你只是最後防線, 非二次驗證. 不確定就放行, 讓規則層的止損管風險."""


@dataclass
class EntrySignals:
    pair: str
    price: float
    oi_div: float              # OI 背離 z-score
    funding_rate: float        # 8h 基準
    taker_z: float
    net_position_skew: float   # -1..+1, 正=散戶多頭擁擠
    vol_struct: float
    entry_score: float         # 融合分
    recent_ret_1h: float       # 最近 1h 漲幅%
    recent_ret_4h: float       # 最近 4h 漲幅%


@dataclass
class Decision:
    allow: bool
    reason: str
    confidence: float
    latency_ms: int
    raw: str = ""
    backend: str = "ollama"


@dataclass
class ExitSignals:
    pair: str
    held_hours: float
    current_profit: float
    oi_div: float
    funding_rate: float
    taker_z: float
    net_position_skew: float
    recent_ret_1h: float


def _fallback_decision(why: str, signals: EntrySignals) -> Decision:
    """Ollama 不可用時的硬規則. 同 prompt 的拒絕條件, 其餘放行."""
    if signals.net_position_skew > 0.5:
        return Decision(False, f"散戶擁擠 skew={signals.net_position_skew:.2f}", 0.6, 0, backend="fallback")
    if abs(signals.funding_rate) > 0.0015:
        return Decision(False, f"funding 極端 {signals.funding_rate*100:+.3f}%", 0.6, 0, backend="fallback")
    if signals.taker_z < -1.5:
        return Decision(False, f"主動賣方壓倒 taker_z={signals.taker_z:.2f}", 0.6, 0, backend="fallback")
    if signals.recent_ret_1h > 0.05:
        return Decision(False, f"短期過熱 1h +{signals.recent_ret_1h*100:.1f}%", 0.6, 0, backend="fallback")
    if signals.oi_div < 0.5:
        return Decision(False, f"OI 背離不足 {signals.oi_div:.2f}", 0.55, 0, backend="fallback")
    return Decision(True, f"規則放行 ({why})", 0.65, 0, backend="fallback")


# Per-pair cooldown: 拒絕過的 pair 在 ttl 內不重問, 直接用上次決策
_decision_cache: dict[str, tuple[float, Decision]] = {}
DECISION_COOLDOWN_SEC = 300  # 5 分鐘


def should_enter(signals: EntrySignals) -> Decision:
    # cooldown check
    cached = _decision_cache.get(signals.pair)
    if cached and (time.time() - cached[0]) < DECISION_COOLDOWN_SEC:
        prev = cached[1]
        prev.latency_ms = 0  # cached
        return prev

    t0 = time.time()
    user_content = (
        f"標的: {signals.pair}\n"
        f"價格: {signals.price}\n"
        f"OI 背離 z: {signals.oi_div:+.2f}\n"
        f"Funding (8h): {signals.funding_rate*100:+.4f}%\n"
        f"Taker z: {signals.taker_z:+.2f}\n"
        f"帳戶淨持倉偏移: {signals.net_position_skew:+.2f} (正=散戶多頭擁擠)\n"
        f"量價結構 z: {signals.vol_struct:+.2f}\n"
        f"融合分: {signals.entry_score:+.2f}\n"
        f"最近 1h 漲幅: {signals.recent_ret_1h:+.2%}\n"
        f"最近 4h 漲幅: {signals.recent_ret_4h:+.2%}\n"
    )
    try:
        content = llm_chat(
            messages=[
                {"role": "system", "content": ENTRY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            json_mode=True,
            temperature=0.3,
            timeout=DECISION_TIMEOUT,
        )
        parsed = json.loads(content)
        latency = int((time.time() - t0) * 1000)
        decision = Decision(
            allow=bool(parsed.get("allow", False)),
            reason=str(parsed.get("reason", ""))[:60],
            confidence=float(parsed.get("confidence", 0.5)),
            latency_ms=latency,
            raw=content,
            backend="openrouter-free",
        )
        _decision_cache[signals.pair] = (time.time(), decision)
        return decision
    except httpx.TimeoutException:
        logger.warning("lana_decision: llm timeout after %.1fs, fallback rules", DECISION_TIMEOUT)
        return _fallback_decision("timeout", signals)
    except Exception as e:
        logger.warning("lana_decision: llm err %s, fallback rules", e)
        return _fallback_decision(str(e)[:40], signals)


def should_hold_on_timestop(signals: ExitSignals) -> Decision:
    """time_stop 觸發前的 sanity check. 回傳 allow 的語意是 'hold (延後出場)'."""
    t0 = time.time()
    content = (
        f"標的: {signals.pair}\n"
        f"持倉時間: {signals.held_hours:.1f}h\n"
        f"當前損益: {signals.current_profit:+.2%}\n"
        f"OI 背離 z: {signals.oi_div:+.2f}\n"
        f"Funding (8h): {signals.funding_rate*100:+.4f}%\n"
        f"Taker z: {signals.taker_z:+.2f}\n"
        f"散戶偏移: {signals.net_position_skew:+.2f}\n"
        f"最近 1h: {signals.recent_ret_1h:+.2%}\n"
    )
    try:
        raw = llm_chat(
            messages=[
                {"role": "system", "content": EXIT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            json_mode=True,
            temperature=0.3,
            timeout=DECISION_TIMEOUT,
        )
        parsed = json.loads(raw or "{}")
        latency = int((time.time() - t0) * 1000)
        return Decision(
            allow=bool(parsed.get("hold", False)),  # allow=hold
            reason=str(parsed.get("reason", ""))[:60],
            confidence=float(parsed.get("confidence", 0.5)),
            latency_ms=latency,
            backend="openrouter-free",
        )
    except Exception as e:
        logger.debug("should_hold_on_timestop err: %s", e)
        return Decision(False, "LLM 不可用, 依規則平倉", 0.5, 0, backend="fallback")


def log_decision(pair: str, decision: Decision, signals: EntrySignals):
    """把決策寫 log 供覆盤."""
    logger.info(
        "[LANA_DECISION] %s allow=%s conf=%.2f backend=%s latency=%dms reason=%s signals=%s",
        pair, decision.allow, decision.confidence, decision.backend,
        decision.latency_ms, decision.reason, asdict(signals),
    )


def push_decision_tg(pair: str, decision: Decision, signals: EntrySignals):
    """qwen advisory 分析推 TG (參考, 不阻擋進場)."""
    verdict_icon = "👍" if decision.allow else "⚠️"
    verdict = "看起來 OK" if decision.allow else "有疑慮"
    conf_pct = int(decision.confidence * 100)
    text = (
        f"🧠 Lana 判讀（參考） {pair}\n\n"
        f"{verdict_icon} LLM 分析：{verdict}（信心 {conf_pct}%）\n"
        f"理由：{decision.reason}\n\n"
        f"當下信號（規則層已放行）\n"
        f"  OI 背離 z  = {signals.oi_div:+.2f}\n"
        f"  Funding   = {signals.funding_rate*100:+.4f}%\n"
        f"  散戶偏移   = {signals.net_position_skew:+.2f}\n"
        f"  Taker z   = {signals.taker_z:+.2f}\n"
        f"  融合分    = {signals.entry_score:+.2f}\n"
        f"  1h / 4h 漲幅 = {signals.recent_ret_1h:+.2%} / {signals.recent_ret_4h:+.2%}\n"
        f"\n進場由規則層決定，LLM 只作觀察備註\n後端: {decision.backend} · {decision.latency_ms}ms"
    )
    tg_send(text)
