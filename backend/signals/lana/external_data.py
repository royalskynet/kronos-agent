"""把 Binance public REST 的 OI / funding / taker 合進 freqtrade dataframe。

freqtrade 從 Bitget 抓的 K 線只含 OHLCV. 我們的 Lana edge 依賴 OI 背離,
因此每個 pair 在 populate_indicators 時到 Binance 抓對應的附加序列.

Binance symbol 映射: Bitget pair "XYZ/USDT:USDT" → Binance "XYZUSDT".
若 Binance 未上該 pair → 各指標 fallback 0 (策略降級為純價量 signal).

每 pair 的 fetch 結果在 memory 快取 TTL 5 分鐘, 避免每 bar 都打 API.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
BITGET = "https://api.bitget.com"
TTL_SEC = 280  # 略小於 5 分鐘避免 race
_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()
_cache_lock = threading.Lock()


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                timeout=10.0,
                headers={"User-Agent": "lana-agent/0.1"},
                transport=httpx.HTTPTransport(retries=2),
            )
        return _client


@dataclass
class _Entry:
    oi: pd.DataFrame = field(default_factory=pd.DataFrame)              # Binance OI hist (Lana core edge)
    funding: pd.DataFrame = field(default_factory=pd.DataFrame)         # Bitget native (fallback Binance)
    taker: pd.DataFrame = field(default_factory=pd.DataFrame)           # Binance takerlongshortRatio
    position_skew: pd.DataFrame = field(default_factory=pd.DataFrame)   # Bitget account-long-short
    fetched_at: float = 0.0
    exists_on_binance: bool = True
    exists_on_bitget: bool = True


_cache: dict[str, _Entry] = {}


BINANCE_1000X_PREFIX = {
    "PEPE", "SHIB", "BONK", "FLOKI", "SATS", "RATS", "LUNC", "CAT", "CHEEMS",
    "WHY", "XEC", "XEN", "MOG", "TURBO", "X",
}


def binance_symbol(pair: str) -> str:
    """Bitget 'XYZ/USDT:USDT' → Binance 符號.

    處理 Binance 的 1000x 系列 (1000PEPEUSDT 等).
    """
    base = pair.split("/")[0]
    if base in BINANCE_1000X_PREFIX:
        return f"1000{base}USDT"
    quote_symbol = pair.split(":")[0].replace("/", "").replace("-", "")
    return quote_symbol


def _fetch_oi_hist(symbol: str) -> pd.DataFrame:
    """最近 500 根 5m OI (約 41h)."""
    r = _http().get(
        f"{FAPI}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": "5m", "limit": 500},
    )
    if r.status_code == 400:
        return pd.DataFrame()  # symbol 不存在
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    return df[["ts", "open_interest"]].sort_values("ts").reset_index(drop=True)


def _fetch_funding(symbol: str) -> pd.DataFrame:
    """Binance funding 歷史 (fallback)."""
    r = _http().get(f"{FAPI}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 100})
    if r.status_code == 400:
        return pd.DataFrame()
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)


def _fetch_bitget_funding(bitget_symbol: str) -> pd.DataFrame:
    """Bitget native funding 歷史 — 實盤交易所這邊的真實 funding.

    回傳最多 20 pages × 20 條 ≈ 400 筆, 足夠 strategy 使用.
    """
    r = _http().get(
        f"{BITGET}/api/v2/mix/market/history-fund-rate",
        params={"symbol": bitget_symbol, "productType": "usdt-futures", "pageSize": 100},
    )
    if r.status_code != 200:
        return pd.DataFrame()
    payload = r.json()
    if payload.get("code") != "00000":
        return pd.DataFrame()
    data = payload.get("data") or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)


def _fetch_bitget_account_ls(bitget_symbol: str) -> pd.DataFrame:
    """Bitget account-long-short 比時序 — period=5m 最多 500 筆."""
    r = _http().get(
        f"{BITGET}/api/v2/mix/market/account-long-short",
        params={"symbol": bitget_symbol, "productType": "usdt-futures", "period": "5m"},
    )
    if r.status_code != 200:
        return pd.DataFrame()
    payload = r.json()
    if payload.get("code") != "00000":
        return pd.DataFrame()
    data = payload.get("data") or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    df["long_account_ratio"] = df["longAccountRatio"].astype(float)
    df["short_account_ratio"] = df["shortAccountRatio"].astype(float)
    # long - short 作為淨持倉偏移指標 (-1 ~ +1)
    df["net_position_skew"] = df["long_account_ratio"] - df["short_account_ratio"]
    return df[["ts", "net_position_skew", "long_account_ratio"]].sort_values("ts").reset_index(drop=True)


def _fetch_taker(symbol: str) -> pd.DataFrame:
    """最近 500 根 5m taker long/short ratio."""
    r = _http().get(
        f"{FAPI}/futures/data/takerlongshortRatio",
        params={"symbol": symbol, "period": "5m", "limit": 500},
    )
    if r.status_code == 400:
        return pd.DataFrame()
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["taker_ls_ratio"] = df["buySellRatio"].astype(float)
    return df[["ts", "taker_ls_ratio"]].sort_values("ts").reset_index(drop=True)


def bitget_symbol(pair: str) -> str:
    """'XYZ/USDT:USDT' → 'XYZUSDT' (Bitget format)."""
    base_quote = pair.split(":")[0]
    return base_quote.replace("/", "").replace("-", "")


def get_externals(pair: str) -> _Entry:
    """回傳 (oi / funding / taker / position_skew) 快取的 entry.

    混合架構:
      - OI 歷史: Binance (Bitget 只有 current snapshot, Lana 核心 edge 要歷史)
      - Funding 歷史: Bitget native (實盤這邊真實) + Binance fallback
      - Taker ratio: Binance (Bitget 無)
      - Position skew: Bitget account-long-short (實盤帳戶持倉)
    """
    b_sym = binance_symbol(pair)
    g_sym = bitget_symbol(pair)
    now = time.time()
    with _cache_lock:
        entry = _cache.get(pair)
        if entry and (now - entry.fetched_at) < TTL_SEC:
            return entry
    try:
        oi = _fetch_oi_hist(b_sym)
        taker = _fetch_taker(b_sym)
        exists_binance = not oi.empty

        # Bitget native 先 try, 失敗時 fallback Binance funding
        funding = _fetch_bitget_funding(g_sym)
        if funding.empty:
            funding = _fetch_funding(b_sym)
        position_skew = _fetch_bitget_account_ls(g_sym)
        exists_bitget = not (funding.empty and position_skew.empty)
    except Exception as e:
        logger.warning("external_data fetch error for %s (%s/%s): %s", pair, b_sym, g_sym, e)
        oi = funding = taker = position_skew = pd.DataFrame()
        exists_binance = exists_bitget = False

    new_entry = _Entry(
        oi=oi,
        funding=funding,
        taker=taker,
        position_skew=position_skew,
        fetched_at=now,
        exists_on_binance=exists_binance,
        exists_on_bitget=exists_bitget,
    )
    with _cache_lock:
        _cache[pair] = new_entry
    if not exists_binance:
        logger.info("external_data: %s (b=%s) not on Binance, OI fallback to zeros", pair, b_sym)
    if not exists_bitget:
        logger.info("external_data: %s (g=%s) Bitget native data empty", pair, g_sym)
    return new_entry


def apply_externals(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """把 OI / funding / taker 以最近匹配方式 merge 進 df.

    freqtrade 的 df 以 `date` 欄保 timestamp (tz-aware UTC).
    我們 merge_asof direction='nearest' 對齊最近一根外部數據.
    """
    e = get_externals(pair)
    ts_col = "date" if "date" in df.columns else None
    if ts_col is None:
        # freqtrade 舊版可能用 index
        df = df.copy()
        df["date"] = df.index
        ts_col = "date"
    df = df.sort_values(ts_col).reset_index(drop=True)

    # 缺少時以 0 建 column (保持 downstream signal 能處理)
    if e.oi.empty:
        df["open_interest"] = 0.0
    else:
        merged = pd.merge_asof(
            df[[ts_col]], e.oi.rename(columns={"ts": ts_col}), on=ts_col, direction="nearest",
            tolerance=pd.Timedelta("10min"),
        )
        df["open_interest"] = merged["open_interest"].fillna(0.0).values

    if e.funding.empty:
        df["funding_rate"] = 0.0
    else:
        merged = pd.merge_asof(
            df[[ts_col]], e.funding.rename(columns={"ts": ts_col}), on=ts_col, direction="backward",
            tolerance=pd.Timedelta("8h"),
        )
        df["funding_rate"] = merged["funding_rate"].fillna(0.0).values

    if e.taker.empty:
        df["taker_ls_ratio"] = 1.0  # 中性
    else:
        merged = pd.merge_asof(
            df[[ts_col]], e.taker.rename(columns={"ts": ts_col}), on=ts_col, direction="nearest",
            tolerance=pd.Timedelta("10min"),
        )
        df["taker_ls_ratio"] = merged["taker_ls_ratio"].fillna(1.0).values

    # Bitget account-long-short 淨持倉偏移 (-1..+1)
    if e.position_skew.empty:
        df["net_position_skew"] = 0.0
    else:
        merged = pd.merge_asof(
            df[[ts_col]], e.position_skew.rename(columns={"ts": ts_col}), on=ts_col, direction="nearest",
            tolerance=pd.Timedelta("10min"),
        )
        df["net_position_skew"] = merged["net_position_skew"].fillna(0.0).values

    return df
