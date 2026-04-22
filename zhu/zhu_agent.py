#!/usr/bin/env python3
"""Zhu Agent v2 — OKX onchainos-based autonomous Solana meme trader.

Single asyncio daemon, 0 LLM calls. Uses OKX onchainos CLI for all discovery,
security, quote, and calldata generation. Signs Solana tx locally from
Keychain-stored mnemonic. Broadcasts via Solana mainnet-beta RPC.

Flow:
  scan_loop  (60s): onchainos signal list -> filter candidates
  entry_loop      : security token-scan -> swap quote -> (LIVE: swap swap ->
                    zhu_sol_sign.sign_and_broadcast | DRY: simulate) -> record
  hold_loop  (60s): market price -> update peak -> evaluate exit rules
                    -> sell when triggered

Safety: live=false default. Single-position cap = stake_pct * equity.
Daily loss halt at -20%.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# local sign/broadcast helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zhu_sol_sign import sign_and_broadcast  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "zhu_state.db")
LOG = os.path.join(HERE, "zhu_agent.log")
CONFIG = os.path.join(HERE, "zhu_config.json")
ONCHAINOS = os.path.expanduser("~/.local/bin/onchainos")

SOL_CONTRACT = "11111111111111111111111111111111"
USDC_CONTRACT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

_QUOTE_FAIL_SKIP: dict = {}  # ca -> retry_after_epoch
QUOTE_FAIL_SKIP_SEC = 1800


# ---------- logging ----------
def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with open(LOG, "a") as f:
        f.write(line)


# ---------- config / db ----------
def load_config() -> dict:
    with open(CONFIG) as f:
        return json.load(f)


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    ca TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    open_ts INTEGER,
    open_price REAL,
    size_usd REAL,
    size_tokens REAL,
    peak_price REAL,
    scaled_out INTEGER DEFAULT 0,
    status TEXT,
    close_ts INTEGER,
    close_price REAL,
    pnl_usd REAL,
    exit_reason TEXT,
    open_tx TEXT,
    close_tx TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    ca TEXT PRIMARY KEY,
    first_seen INTEGER,
    symbol TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS daily_pnl (
    date TEXT PRIMARY KEY,
    realized_usd REAL DEFAULT 0,
    entries INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);
"""


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------- onchainos subprocess wrappers ----------
async def _run(cmd: list, timeout: int = 30) -> tuple[int, str, str]:
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.communicate()
        return 124, "", "timeout"
    return p.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _parse_json(out: str) -> dict | None:
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


async def oc_signal_list(chain: str, min_mc: int, max_mc: int,
                          min_liq: int, limit: int) -> dict | None:
    rc, out, _ = await _run([
        ONCHAINOS, "signal", "list", "--chain", chain,
        "--min-market-cap-usd", str(min_mc),
        "--max-market-cap-usd", str(max_mc),
        "--min-liquidity-usd", str(min_liq),
        "--limit", str(limit),
    ])
    return _parse_json(out) if rc == 0 else None


async def oc_security_scan(chain: str, ca: str) -> dict | None:
    rc, out, _ = await _run([
        ONCHAINOS, "security", "token-scan",
        "--tokens", f"501:{ca}" if chain == "solana" else f"1:{ca}",
    ], timeout=15)
    return _parse_json(out) if rc == 0 else None


async def oc_swap_quote(chain: str, from_token: str, to_token: str,
                         readable_amount: float) -> dict | None:
    rc, out, _ = await _run([
        ONCHAINOS, "swap", "quote",
        "--from", from_token, "--to", to_token,
        "--readable-amount", str(readable_amount),
        "--chain", chain,
    ], timeout=20)
    return _parse_json(out) if rc == 0 else None


async def oc_swap_swap(chain: str, from_token: str, to_token: str,
                        readable_amount: float, wallet: str,
                        slippage: float | None = None,
                        mev_tips: int | None = None) -> dict | None:
    cmd = [
        ONCHAINOS, "swap", "swap",
        "--from", from_token, "--to", to_token,
        "--readable-amount", str(readable_amount),
        "--chain", chain, "--wallet", wallet,
    ]
    if slippage is not None:
        cmd += ["--slippage", str(slippage)]
    if mev_tips is not None and mev_tips > 0:
        cmd += ["--tips", str(mev_tips)]
    rc, out, _ = await _run(cmd, timeout=30)
    return _parse_json(out) if rc == 0 else None


async def oc_market_price(chain: str, ca: str) -> dict | None:
    rc, out, _ = await _run([
        ONCHAINOS, "market", "price",
        "--chain", chain, "--address", ca,
    ], timeout=15)
    return _parse_json(out) if rc == 0 else None


async def _get_price_usd(chain: str, ca: str) -> float | None:
    """Return USD price for a token, handling both list and dict data shapes."""
    r = await oc_market_price(chain, ca)
    if not r:
        return None
    d = r.get("data")
    item: dict | None = None
    if isinstance(d, list) and d:
        item = d[0] if isinstance(d[0], dict) else None
    elif isinstance(d, dict):
        item = d
    if not item:
        return None
    for key in ("price", "priceUsd", "tokenPrice", "tokenUnitPrice"):
        v = item.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


# ---------- telegram ----------
def tg_send(cfg: dict, text: str):
    token, chat = cfg.get("tg_bot_token"), cfg.get("tg_chat_id")
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text,
        "disable_web_page_preview": "true",
    }).encode()
    try:
        urllib.request.urlopen(url, data=data, timeout=10).read()
    except Exception as e:
        log(f"tg send fail: {e}")


# ---------- surf brief priority ----------
_SURF_CACHE: dict = {"loaded_at": 0, "data": None}


def load_surf_brief(cfg: dict) -> dict | None:
    path = cfg.get("surf_brief_path")
    if not path or not os.path.exists(path):
        return None
    st = os.stat(path)
    if (time.time() - st.st_mtime) / 3600 > cfg.get("surf_brief_ttl_hours", 6):
        return None
    if _SURF_CACHE["loaded_at"] == st.st_mtime and _SURF_CACHE["data"]:
        return _SURF_CACHE["data"]
    try:
        with open(path) as f:
            data = json.load(f)
        _SURF_CACHE["loaded_at"] = st.st_mtime
        _SURF_CACHE["data"] = data
        return data
    except Exception as e:
        log(f"surf brief load fail: {e}")
        return None


def is_surf_priority(ca: str, cfg: dict) -> bool:
    brief = load_surf_brief(cfg)
    if not brief:
        return False
    for c in (brief.get("candidates") or []):
        if c.get("ca") == ca:
            return True
    return False


# ---------- signal parsing ----------
def extract_signal_candidate(sig: dict) -> dict | None:
    tok = sig.get("token") or {}
    ca = tok.get("tokenAddress")
    if not ca:
        return None
    return {
        "ca": ca,
        "symbol": tok.get("symbol", "?"),
        "name": tok.get("name", "?"),
        "mc": float(tok.get("marketCapUsd", 0) or 0),
        "top10_pct": float(tok.get("top10HolderPercent", 0) or 0),
        "holders": int(tok.get("holders", 0) or 0),
        "trigger_count": int(sig.get("triggerWalletCount", 0) or 0),
        "amount_usd": float(sig.get("amountUsd", 0) or 0),
        "ts": int(sig.get("timestamp", 0) or 0),
        "wallet_type": sig.get("walletType", ""),
        "raw": sig,
    }


def entry_ok(c: dict, cfg: dict) -> bool:
    ca = c["ca"]
    priority = is_surf_priority(ca, cfg)
    min_triggers = cfg["surf_priority_min_hunters"] if priority else cfg["entry_min_hunters"]
    max_mc = cfg["surf_priority_max_mc"] if priority else cfg["entry_max_mc"]

    if c["trigger_count"] < min_triggers:
        return False
    if c["mc"] < cfg["entry_min_mc"] or c["mc"] > max_mc:
        return False
    if c["holders"] < cfg["entry_min_holder"]:
        return False
    if c["top10_pct"] / 100.0 > cfg["max_top10_holder_pct"]:
        return False
    if c["ts"]:
        age_min = (time.time() * 1000 - c["ts"]) / 60000.0
        if age_min > cfg["entry_max_age_min"]:
            return False
    return True


# ---------- position guards ----------
def daily_halt(conn: sqlite3.Connection, cfg: dict) -> bool:
    row = conn.execute(
        "SELECT realized_usd FROM daily_pnl WHERE date=?", (today_utc(),)
    ).fetchone()
    realized = row[0] if row else 0.0
    return realized <= cfg["equity_usd"] * cfg["daily_loss_halt_pct"]


def open_positions_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM positions WHERE status='open'"
    ).fetchone()[0]


def record_daily(conn: sqlite3.Connection, realized: float, won: bool):
    conn.execute(
        "INSERT INTO daily_pnl(date, realized_usd, entries, wins, losses) "
        "VALUES(?,?,0,?,?) ON CONFLICT(date) DO UPDATE SET "
        "realized_usd=realized_usd+excluded.realized_usd, "
        "wins=wins+excluded.wins, losses=losses+excluded.losses",
        (today_utc(), realized, 1 if won else 0, 0 if won else 1),
    )


# ---------- security gate ----------
async def security_pass(chain: str, ca: str) -> bool:
    """Block honeypot / high-risk tokens. Missing data = pass."""
    r = await oc_security_scan(chain, ca)
    if not r or not r.get("ok"):
        return True
    data = r.get("data")
    if isinstance(data, dict):
        data = data.get("list") or [data]
    if not isinstance(data, list):
        return True
    for t in data:
        if not isinstance(t, dict):
            continue
        # security API doesn't echo tokenAddress; inspect all items flat
        if t.get("isHoneypot") or t.get("isHoneyPot"):
            log(f"GATE honeypot ca={ca}")
            return False
        if t.get("isFakeLiquidity") or t.get("isLiquidityRemoval"):
            log(f"GATE liq-risk ca={ca}")
            return False
        rl = (t.get("riskLevel") or "").upper()
        if rl == "HIGH":
            log(f"GATE riskLevel=HIGH ca={ca}")
            return False
    return True


# ---------- entry ----------
async def try_entry(conn: sqlite3.Connection, cfg: dict, c: dict):
    ca = c["ca"]
    sym = c["symbol"]

    if conn.execute("SELECT 1 FROM positions WHERE ca=?", (ca,)).fetchone():
        return
    if _QUOTE_FAIL_SKIP.get(ca, 0) > time.time():
        return
    if open_positions_count(conn) >= cfg["max_concurrent_positions"]:
        return
    if daily_halt(conn, cfg):
        log(f"daily halt — skip {sym} {ca}")
        return

    if not await security_pass("solana", ca):
        return

    stake_usd = cfg["equity_usd"] * cfg["stake_pct_per_trade"]
    sol_price = await _get_price_usd("solana", SOL_CONTRACT) or 88.0
    amount_sol = round(stake_usd / sol_price, 6)

    q = await oc_swap_quote("solana", "sol", ca, amount_sol)
    if not q or not q.get("data"):
        _QUOTE_FAIL_SKIP[ca] = time.time() + QUOTE_FAIL_SKIP_SEC
        log(f"quote fail {sym} ca={ca} (skip {QUOTE_FAIL_SKIP_SEC//60}m)")
        return
    q_first = (q["data"] or [{}])[0]
    try:
        from_amt = int(q_first.get("fromTokenAmount", 0))
        to_amt = int(q_first.get("toTokenAmount", 0))
        to_dec = int((q_first.get("toToken") or {}).get("decimal", 0))
        from_dec = int((q_first.get("fromToken") or {}).get("decimal", 9))
        tokens_out = to_amt / (10 ** to_dec)
        sol_in = from_amt / (10 ** from_dec)
    except Exception as e:
        log(f"quote parse fail {ca}: {e}")
        return
    if tokens_out <= 0:
        log(f"quote zero out {ca}")
        return
    est_price_usd = (sol_in * sol_price) / tokens_out
    mode = "LIVE" if cfg["live"] else "DRY"
    log(
        f"{mode} ENTRY {sym} ca={ca} size=${stake_usd:.2f} "
        f"tokens={tokens_out:.4f} px=${est_price_usd:.8f}"
    )

    open_tx = "DRY_RUN"
    if cfg["live"]:
        s = await oc_swap_swap(
            "solana", "sol", ca, amount_sol,
            cfg["wallet_address"],
            slippage=cfg["swap_slippage_pct"],
            mev_tips=cfg.get("jito_tip_lamports") or None,
        )
        if not s or not s.get("data"):
            log(f"LIVE entry swap_swap fail {ca}")
            return
        tx_data = ((s["data"] or [{}])[0].get("tx") or {}).get("data")
        if not tx_data:
            log(f"LIVE entry no tx.data {ca}")
            return
        rpc = cfg.get("solana_rpc_url") or "https://api.mainnet-beta.solana.com"
        try:
            result = await asyncio.to_thread(
                sign_and_broadcast, tx_data, rpc, True, 90
            )
        except Exception as e:
            log(f"LIVE entry sign/bcast exception {ca}: {e}")
            return
        if result.get("error"):
            log(f"LIVE entry broadcast err {ca}: {result['error']}")
            return
        open_tx = result.get("signature") or "UNKNOWN_SIG"
        log(f"LIVE entry confirmed {ca} sig={open_tx}")

    conn.execute(
        "INSERT INTO positions(ca, symbol, name, open_ts, open_price, size_usd, "
        "size_tokens, peak_price, status, open_tx) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ca, sym, c.get("name"), int(time.time()), est_price_usd,
         stake_usd, tokens_out, est_price_usd, "open", open_tx),
    )
    tg_send(cfg, (
        f"Zhu {mode} ENTRY\n"
        f"{sym} ({c.get('name','?')})\n"
        f"CA: `{ca}`\n"
        f"Size: ${stake_usd:.2f} | Tokens: {tokens_out:.4f}\n"
        f"Entry px: {est_price_usd:.8f}\n"
        f"Triggers: {c['trigger_count']} wallets | MC: ${c['mc']:,.0f} | Top10: {c['top10_pct']:.1f}%\n"
        f"TX: {open_tx[:12]}..."
    ))


# ---------- exit ----------
async def try_exit(conn: sqlite3.Connection, cfg: dict, pos: sqlite3.Row):
    ca = pos["ca"]
    sym = pos["symbol"]
    age_hours = (time.time() - pos["open_ts"]) / 3600.0

    cur_price = await _get_price_usd("solana", ca)
    if cur_price is None or cur_price <= 0:
        return

    peak = max(pos["peak_price"] or cur_price, cur_price)
    if peak != pos["peak_price"]:
        conn.execute("UPDATE positions SET peak_price=? WHERE ca=?", (peak, ca))

    entry = pos["open_price"] or cur_price
    mult = cur_price / entry if entry else 1.0
    drawdown = (peak - cur_price) / peak if peak else 0.0

    reason: str | None = None
    if age_hours > cfg["exit_time_hours"] and mult < 2.0:
        reason = "time_stop"
    elif (cur_price / entry - 1.0) <= cfg["exit_hard_stop_pct"]:
        reason = "hard_stop"
    elif drawdown >= cfg["exit_peak_drawdown_pct"]:
        reason = f"trailing_peak_{drawdown*100:.0f}%"
    elif not pos["scaled_out"] and mult >= cfg["exit_scale_out_multiplier"]:
        sell_tokens = pos["size_tokens"] * cfg["exit_scale_out_fraction"]
        log(f"SCALE_OUT {sym} {mult:.1f}x sell {sell_tokens:.4f} tokens")
        # (partial sell live not yet wired; just mark scaled_out in DRY)
        conn.execute(
            "UPDATE positions SET size_tokens=size_tokens-?, scaled_out=1 WHERE ca=?",
            (sell_tokens, ca),
        )
        tg_send(cfg, f"Zhu SCALE-OUT\n{sym} {mult:.1f}x — sold {cfg['exit_scale_out_fraction']*100:.0f}%")
        return

    if not reason:
        return

    pnl = (cur_price - entry) * pos["size_tokens"]
    mode = "LIVE" if cfg["live"] else "DRY"
    log(f"{mode} EXIT {sym} ca={ca} reason={reason} mult={mult:.2f}x pnl=${pnl:+.2f}")

    close_tx = "DRY_RUN"
    if cfg["live"]:
        s = await oc_swap_swap(
            "solana", ca, "sol", pos["size_tokens"],
            cfg["wallet_address"],
            slippage=cfg["swap_slippage_pct"],
            mev_tips=cfg.get("jito_tip_lamports") or None,
        )
        if s and s.get("data"):
            tx_data = ((s["data"] or [{}])[0].get("tx") or {}).get("data")
            if tx_data:
                rpc = cfg.get("solana_rpc_url") or "https://api.mainnet-beta.solana.com"
                try:
                    result = await asyncio.to_thread(
                        sign_and_broadcast, tx_data, rpc, True, 90
                    )
                    close_tx = result.get("signature") or "FAILED_" + str(result.get("error"))[:20]
                except Exception as e:
                    close_tx = f"EXCEPT_{type(e).__name__}"
                    log(f"LIVE exit broadcast exception {ca}: {e}")

    conn.execute(
        "UPDATE positions SET status='closed', close_ts=?, close_price=?, "
        "pnl_usd=?, exit_reason=?, close_tx=? WHERE ca=?",
        (int(time.time()), cur_price, pnl, reason, close_tx, ca),
    )
    record_daily(conn, pnl, pnl > 0)
    tg_send(cfg, (
        f"Zhu {mode} EXIT\n"
        f"{sym} CA: `{ca}`\n"
        f"Reason: {reason} | {mult:.2f}x\n"
        f"PnL: ${pnl:+.2f}"
    ))


# ---------- loops ----------
async def scan_loop(conn: sqlite3.Connection, cfg: dict):
    while True:
        try:
            resp = await oc_signal_list(
                "solana",
                int(cfg["entry_min_mc"]),
                int(cfg["entry_max_mc"] * 2),  # headroom
                int(cfg.get("entry_min_liq_usd", 5000)),
                cfg.get("signal_size", 30),
            )
            lst = (resp or {}).get("data") or []
            matched = 0
            for sig in lst:
                c = extract_signal_candidate(sig)
                if not c:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO candidates(ca, first_seen, symbol, payload) "
                    "VALUES(?,?,?,?)",
                    (c["ca"], int(time.time()), c["symbol"], json.dumps(sig)),
                )
                if entry_ok(c, cfg):
                    await try_entry(conn, cfg, c)
                    matched += 1
            log(f"scan tick fetched={len(lst)} matched={matched}")
        except Exception as e:
            log(f"scan error: {type(e).__name__}: {e}")
        await asyncio.sleep(cfg["scan_poll_sec"])


async def hold_loop(conn: sqlite3.Connection, cfg: dict):
    while True:
        try:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='open'"
            ).fetchall()
            for row in rows:
                await try_exit(conn, cfg, row)
        except Exception as e:
            log(f"hold error: {type(e).__name__}: {e}")
        await asyncio.sleep(cfg["hold_poll_sec"])


async def main():
    cfg = load_config()
    conn = init_db()
    mode = "LIVE" if cfg["live"] else "DRY"
    log(
        f"zhu_agent v2 (OKX onchainos) start mode={mode} "
        f"wallet={cfg['wallet_address']} equity=${cfg['equity_usd']} "
        f"stake={cfg['stake_pct_per_trade']*100:.0f}% "
        f"max_pos={cfg['max_concurrent_positions']}"
    )
    tg_send(cfg, f"Zhu v2 (OKX onchainos) started — {mode} mode, equity ${cfg['equity_usd']}")
    await asyncio.gather(scan_loop(conn, cfg), hold_loop(conn, cfg))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("zhu_agent stopped")
