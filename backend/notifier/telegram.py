from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any

from ..config import read_telegram_settings


def tg_send(text: str, *, silent: bool = False) -> None:
    cfg = read_telegram_settings()
    if not cfg.get("enabled"):
        return
    token = str(cfg.get("bot_token") or "").strip()
    chat = str(cfg.get("chat_id") or "").strip()
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text[:4096],
        "disable_web_page_preview": "true",
        "disable_notification": "true" if silent else "false",
    }
    data = urllib.parse.urlencode(payload).encode()
    try:
        urllib.request.urlopen(url, data=data, timeout=10).read()
    except Exception:
        pass


def tg_notify_decision(decision: dict[str, Any], *, dry_run: bool = False) -> None:
    actions: list[dict[str, Any]] = decision.get("actions", [])
    warnings: list[str] = decision.get("warnings", [])
    prefix = "[DRY] " if dry_run else ""
    lines: list[str] = []

    open_acts = [a for a in actions if a.get("type") == "open"]
    close_acts = [a for a in actions if a.get("type") in {"close", "reduce"}]
    breaker_acts = [a for a in actions if a.get("type") == "circuit_breaker"]

    for a in open_acts:
        side = str(a.get("side") or "").upper()
        sym = a.get("symbol", "")
        notional = a.get("notionalUsd") or 0
        conf = a.get("confidence") or 0
        reason = str(a.get("reason") or "")[:120]
        lines.append(f"{prefix}OPEN {side} {sym} ${notional:.0f} conf={conf}\n{reason}")

    for a in close_acts:
        sym = a.get("symbol", "")
        pnl = a.get("realizedPnlUsd") or 0
        reason = str(a.get("reason") or "")[:80]
        sign = "+" if pnl >= 0 else ""
        lines.append(f"{prefix}CLOSE {sym} PnL={sign}{pnl:.2f} — {reason}")

    for a in breaker_acts:
        lines.append(f"{prefix}CIRCUIT BREAKER fired — all positions closed")

    if warnings:
        for w in warnings[:3]:
            lines.append(f"WARN: {w[:120]}")

    if lines:
        tg_send("\n\n".join(lines))
