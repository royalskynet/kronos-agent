"""Bridge to TauricResearch/TradingAgents (P2+).

TradingAgents 是 LangGraph-based 多 agent 框架，我們 fork 進 vendored/ 並只啟用：
  - SentimentAnalyst (我們已有 NarrativeAgent 取代)
  - BullResearcher / BearResearcher
  - RiskManager
  - Trader (最終決策)

本檔在 P1 是 stub；P2 升級時：
  git submodule add https://github.com/TauricResearch/TradingAgents vendored/TradingAgents
然後 import 並 wire 進策略。

**硬規則**：只接受 signal_vector + narrative_summary 兩種輸入；永遠不看 user 推文 / 倉位。
"""
from __future__ import annotations

from typing import Any


def run_debate(signal_vector: dict, narrative: str) -> dict:
    """P2 實作：調 TradingAgents LangGraph。"""
    return {
        "bull_case": "(P2 pending)",
        "bear_case": "(P2 pending)",
        "risk_flag": False,
        "score_adjust": 0.0,
    }
