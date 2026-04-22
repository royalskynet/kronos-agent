"""RiskSanity (P2+) — 進場前 final sanity check (LLM, Opus)。

freqtrade 的 Protections 已處理硬規則熔斷。本模組只補 LLM sanity check：
  - 新聞面是否有意外風險？(黑天鵝、監管消息)
  - 槓桿和倉位是否與最近波動率匹配？
  - 是否在已知爆倉週期內？

回傳 veto=True 即否決進場，freqtrade 走 confirm_trade_entry() 拒收。
"""
from __future__ import annotations


def final_sanity_check(pair: str, stake: float, leverage: float, narrative: dict) -> tuple[bool, str]:
    """回傳 (allow, reason)。P1 default allow。"""
    # P1 stub
    return True, "P1: sanity check skipped"
