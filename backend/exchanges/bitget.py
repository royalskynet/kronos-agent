from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from ..config import read_live_trading_config, read_network_settings
from ..http_client import cached_get_json, request_json
from ..utils import clamp, now_iso, num
from .base import ExchangeGateway
from .catalog import exchange_config

_PRODUCT_TYPE = "USDT-FUTURES"


class BitgetGateway(ExchangeGateway):
    exchange_id = "bitget"
    display_name = "Bitget"
    market_label = "USDT perpetual futures"
    default_backdrop_symbol = "BTCUSDT"
    public_base_url = exchange_config("bitget")["defaultBaseUrl"]
    symbol_pattern = re.compile(r"^[A-Z0-9]+USDT$")
    interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "12h": "12H",
    }

    def candidate_symbol_hint(self) -> str:
        return "Bitget USDT perpetual futures symbols, e.g. BTCUSDT"

    def normalize_symbol(self, symbol: str) -> str:
        s = str(symbol or "").strip().upper()
        s = s.replace("-", "").replace("_", "").replace("/", "")
        if not s.endswith("USDT"):
            s = s + "USDT"
        return s

    def validate_symbol(self, symbol: str) -> bool:
        return bool(self.symbol_pattern.fullmatch(self.normalize_symbol(symbol)))

    def base_asset_from_symbol(self, symbol: str) -> str:
        normalized = self.normalize_symbol(symbol)
        return normalized[:-4] if normalized.endswith("USDT") else normalized

    def _cache_policy_for_kline_interval(self, interval: str) -> tuple[int, int]:
        interval = str(interval or "").lower()
        if interval == "1m":
            return 20, 60 * 60
        if interval == "5m":
            return 30, 2 * 60 * 60
        if interval == "15m":
            return 60, 3 * 60 * 60
        if interval == "1h":
            return 5 * 60, 12 * 60 * 60
        if interval == "4h":
            return 15 * 60, 48 * 60 * 60
        return 60, 6 * 60 * 60

    def _timestamp(self) -> str:
        return str(int(datetime.now(timezone.utc).timestamp() * 1000))

    def _query_string(self, params: dict[str, Any] | None = None) -> str:
        filtered = {k: v for k, v in (params or {}).items() if v not in (None, "", [], {})}
        return urlencode(filtered)

    def _query(self, base_url: str, endpoint: str, params: dict[str, Any] | None = None) -> str:
        qs = self._query_string(params)
        if not qs:
            return f"{base_url.rstrip('/')}{endpoint}"
        return f"{base_url.rstrip('/')}{endpoint}?{qs}"

    def _bitget_data(self, payload: Any, *, endpoint: str) -> Any:
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected Bitget response for {endpoint}.")
        code = str(payload.get("code") or "")
        if code not in {"", "00000"}:
            raise ValueError(f"Bitget {endpoint} failed: code={code} msg={payload.get('msg') or ''}")
        return payload.get("data")

    def _public_get_data(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        namespace: str,
        ttl_seconds: int,
        max_stale_seconds: int,
    ) -> Any:
        network_settings = read_network_settings()
        url = self._query(self.public_base_url, endpoint, params)
        payload = cached_get_json(
            url,
            namespace=f"bitget_{namespace}",
            ttl_seconds=ttl_seconds,
            max_stale_seconds=max_stale_seconds,
            timeout_seconds=45,
            network_settings=network_settings,
        )
        return self._bitget_data(payload, endpoint=endpoint)

    def resolved_base_url(self, config: dict[str, Any]) -> str:
        base = str(config.get("baseUrl") or self.public_base_url).strip().rstrip("/")
        return base or self.public_base_url.rstrip("/")

    def _signed_request_json(
        self,
        config: dict[str, Any],
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | list | None = None,
    ) -> Any:
        network_settings = read_network_settings()
        method_upper = method.upper()
        qs = self._query_string(params)
        request_path = endpoint if not qs else f"{endpoint}?{qs}"
        url = f"{self.resolved_base_url(config)}{request_path}"
        body_text = "" if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        timestamp = self._timestamp()
        prehash = f"{timestamp}{method_upper}{request_path}{body_text}"
        signature = base64.b64encode(
            hmac.new(
                str(config["apiSecret"]).encode("utf-8"),
                prehash.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        headers = {
            "ACCESS-KEY": str(config["apiKey"]),
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": str(config["apiPassphrase"]),
            "Content-Type": "application/json",
            "locale": "en-US",
        }
        payload = request_json(
            method_upper,
            url,
            headers=headers,
            payload=body_text if body is not None else None,
            timeout_seconds=45,
            network_settings=network_settings,
        )
        return self._bitget_data(payload, endpoint=endpoint)

    def _contracts(self) -> list[dict[str, Any]]:
        data = self._public_get_data(
            "/api/v2/mix/market/contracts",
            {"productType": _PRODUCT_TYPE},
            namespace="contracts_usdt",
            ttl_seconds=6 * 60 * 60,
            max_stale_seconds=7 * 24 * 60 * 60,
        )
        return data if isinstance(data, list) else []

    def _symbol_info(self, symbol: str) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        for item in self._contracts():
            if self.normalize_symbol(item.get("symbol")) == normalized:
                return item
        raise ValueError(f"{self.display_name} symbol not found: {normalized}")

    def _step_precision(self, step_size: str | float | None) -> int:
        text = str(step_size or "1")
        if "." not in text:
            return 0
        return len(text.rstrip("0").split(".")[1])

    def _round_down_to_step(self, value: float, step_size: float, precision: int) -> float:
        units = int(value / step_size)
        return round(units * step_size, precision)

    def _round_to_step(self, value: float, step_size: float, precision: int) -> float:
        units = round(value / step_size)
        return round(units * step_size, precision)

    def _map_ticker_row(self, row: dict[str, Any]) -> dict[str, Any]:
        symbol = self.normalize_symbol(row.get("symbol") or "")
        last = num(row.get("lastPr")) or num(row.get("last")) or 0
        open24 = num(row.get("open24h")) or num(row.get("openUtc0")) or 0
        quote_vol = num(row.get("quoteVolume")) or num(row.get("usdtVolume")) or 0
        base_vol = num(row.get("baseVolume")) or num(row.get("baseVol")) or 0
        change_pct = ((last - open24) / open24 * 100) if last > 0 and open24 > 0 else 0
        return {
            "symbol": symbol,
            "lastPrice": last,
            "highPrice": num(row.get("high24h")) or last,
            "lowPrice": num(row.get("low24h")) or last,
            "priceChangePercent": change_pct,
            "quoteVolume": quote_vol,
            "baseVolume": base_vol,
            "ts": row.get("ts"),
            "raw": row,
        }

    def fetch_all_tickers_24h(self) -> list[dict[str, Any]]:
        data = self._public_get_data(
            "/api/v2/mix/market/tickers",
            {"productType": _PRODUCT_TYPE},
            namespace="market_tickers",
            ttl_seconds=60,
            max_stale_seconds=45 * 60,
        )
        rows = data if isinstance(data, list) else []
        return [self._map_ticker_row(r) for r in rows if isinstance(r, dict)]

    def fetch_all_premium_index(self) -> list[dict[str, Any]]:
        try:
            data = self._public_get_data(
                "/api/v2/mix/market/tickers",
                {"productType": _PRODUCT_TYPE},
                namespace="market_tickers_premium",
                ttl_seconds=60,
                max_stale_seconds=45 * 60,
            )
        except Exception:
            return []
        rows = data if isinstance(data, list) else []
        return [
            {
                "symbol": self.normalize_symbol(r.get("symbol") or ""),
                "markPrice": num(r.get("markPrice")) or num(r.get("lastPr")),
                "lastFundingRate": num(r.get("fundingRate")),
                "fundingRate": num(r.get("fundingRate")),
                "raw": r,
            }
            for r in rows
            if isinstance(r, dict) and r.get("symbol")
        ]

    def fetch_ticker_24h(self, symbol: str) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        data = self._public_get_data(
            "/api/v2/mix/market/ticker",
            {"symbol": normalized, "productType": _PRODUCT_TYPE},
            namespace=f"ticker_{normalized}",
            ttl_seconds=20,
            max_stale_seconds=30 * 60,
        )
        row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        return self._map_ticker_row(row)

    def fetch_premium(self, symbol: str) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        funding_row: dict[str, Any] = {}
        try:
            data = self._public_get_data(
                "/api/v2/mix/market/current-fund-rate",
                {"symbol": normalized, "productType": _PRODUCT_TYPE},
                namespace=f"funding_{normalized}",
                ttl_seconds=60,
                max_stale_seconds=12 * 60 * 60,
            )
            if isinstance(data, list) and data:
                funding_row = data[0]
            elif isinstance(data, dict):
                funding_row = data
        except Exception:
            pass
        ticker = self.fetch_ticker_24h(normalized)
        funding_rate = num(funding_row.get("fundingRate"))
        mark_price = num(funding_row.get("markPrice")) or ticker.get("lastPrice")
        return {
            "symbol": normalized,
            "markPrice": mark_price,
            "lastFundingRate": funding_rate,
            "fundingRate": funding_rate,
            "fundingPct": (funding_rate or 0) * 100,
            "nextFundingTime": funding_row.get("nextFundingTime"),
            "raw": {"funding": funding_row},
        }

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        normalized = self.normalize_symbol(symbol)
        granularity = self.interval_map.get(str(interval or "").lower())
        if not granularity:
            raise ValueError(f"Unsupported Bitget kline interval: {interval}")
        ttl_seconds, max_stale_seconds = self._cache_policy_for_kline_interval(interval)
        data = self._public_get_data(
            "/api/v2/mix/market/candles",
            {"symbol": normalized, "productType": _PRODUCT_TYPE, "granularity": granularity, "limit": int(clamp(limit, 1, 300))},
            namespace=f"candles_{normalized}_{granularity}",
            ttl_seconds=ttl_seconds,
            max_stale_seconds=max_stale_seconds,
        )
        rows = data if isinstance(data, list) else []
        parsed: list[dict[str, Any]] = []
        for row in reversed(rows):
            if not isinstance(row, list) or len(row) < 5:
                continue
            close_val = num(row[4])
            if close_val is None:
                continue
            parsed.append({
                "openTime": int(num(row[0]) or 0),
                "open": num(row[1]),
                "high": num(row[2]),
                "low": num(row[3]),
                "close": close_val,
                "volume": num(row[5]),
                "closeTime": int(num(row[0]) or 0),
                "quoteVolume": num(row[6]) if len(row) > 6 else None,
            })
        return parsed

    def live_execution_status(
        self,
        live_config: dict[str, Any] | None = None,
        trading_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        live_config = live_config or read_live_trading_config()
        issues = []
        if not live_config.get("apiKey"):
            issues.append("Bitget API key is missing.")
        if not live_config.get("apiSecret"):
            issues.append("Bitget API secret is missing.")
        if not live_config.get("apiPassphrase"):
            issues.append("Bitget API passphrase is missing.")
        can_sync = not issues
        can_execute = can_sync and live_config.get("enabled") and not live_config.get("dryRun")
        return {
            "configEnabled": live_config.get("enabled") is True,
            "dryRun": live_config.get("dryRun") is True,
            "armed": can_execute,
            "canSync": can_sync,
            "canExecute": can_execute,
            "issues": issues,
            "baseUrl": self.resolved_base_url(live_config),
            "exchange": self.exchange_id,
        }

    def normalize_quantity(
        self,
        config: dict[str, Any],
        symbol: str,
        *,
        reference_price: float | None = None,
        quantity: float | None = None,
        notional_usd: float | None = None,
    ) -> float:
        info = self._symbol_info(symbol)
        size_multiplier = num(info.get("sizeMultiplier")) or 1
        min_trade_num = num(info.get("minTradeNum")) or size_multiplier
        precision = self._step_precision(info.get("sizeMultiplier"))
        raw_qty = quantity
        if raw_qty is None:
            price = reference_price or 1
            raw_qty = (notional_usd or 0) / price if price > 0 else 0
        normalized = self._round_down_to_step(max(0.0, float(raw_qty or 0)), size_multiplier, precision)
        if normalized < min_trade_num:
            raise ValueError(f"Quantity for {self.normalize_symbol(symbol)} is below Bitget minimum.")
        return normalized

    def normalize_price(self, config: dict[str, Any], symbol: str, price: float) -> float:
        info = self._symbol_info(symbol)
        tick = num(info.get("priceEndStep")) or num(info.get("pricePlace"))
        if not tick or tick <= 0:
            return float(price)
        precision = self._step_precision(info.get("priceEndStep"))
        return self._round_to_step(float(price), tick, precision)

    def apply_symbol_settings(self, config: dict[str, Any], symbol: str) -> None:
        normalized = self.normalize_symbol(symbol)
        leverage = str(int(clamp(config.get("defaultLeverage"), 1, 125)))
        margin_mode = "isolated" if str(config.get("marginType") or "crossed").lower() == "isolated" else "crossed"
        self._signed_request_json(
            config, "POST", "/api/v2/mix/account/set-leverage",
            body={"symbol": normalized, "productType": _PRODUCT_TYPE, "marginCoin": "USDT", "leverage": leverage},
        )
        self._signed_request_json(
            config, "POST", "/api/v2/mix/account/set-margin-mode",
            body={"symbol": normalized, "productType": _PRODUCT_TYPE, "marginCoin": "USDT", "marginMode": margin_mode},
        )

    def fetch_account_snapshot(self, config: dict[str, Any], session_started_at: str | None = None) -> dict[str, Any]:
        account_data = self._signed_request_json(
            config, "GET", "/api/v2/mix/account/accounts",
            params={"productType": _PRODUCT_TYPE},
        )
        positions_data = self._signed_request_json(
            config, "GET", "/api/v2/mix/position/all-position",
            params={"productType": _PRODUCT_TYPE, "marginCoin": "USDT"},
        )
        accounts = account_data if isinstance(account_data, list) else ([account_data] if isinstance(account_data, dict) else [])
        usdt_account = next((a for a in accounts if str(a.get("marginCoin") or "").upper() == "USDT"), accounts[0] if accounts else {})

        open_positions: list[dict[str, Any]] = []
        unrealized_pnl = 0.0
        for row in (positions_data if isinstance(positions_data, list) else []):
            total = num(row.get("total"))
            if total is None or abs(total) <= 1e-9:
                continue
            symbol = self.normalize_symbol(row.get("symbol") or "")
            hold_side = str(row.get("holdSide") or "").lower()
            side = "long" if hold_side == "long" else "short"
            quantity = abs(total)
            entry_price = num(row.get("openPriceAvg")) or num(row.get("averageOpenPrice")) or 0
            mark_price = num(row.get("markPrice")) or entry_price
            notional = abs(quantity * mark_price)
            upl = num(row.get("unrealizedPL")) or num(row.get("unrealisedPl")) or 0
            unrealized_pnl += upl
            open_positions.append({
                "id": f"live-{symbol}-{hold_side}",
                "symbol": symbol,
                "baseAsset": self.base_asset_from_symbol(symbol),
                "side": side,
                "quantity": quantity,
                "initialQuantity": quantity,
                "entryPrice": entry_price,
                "notionalUsd": notional,
                "initialNotionalUsd": notional,
                "stopLoss": None,
                "takeProfit": None,
                "lastMarkPrice": mark_price,
                "lastMarkTime": now_iso(),
                "leverage": num(row.get("leverage")) or 1,
                "status": "open",
                "openedAt": None,
                "updatedAt": now_iso(),
                "source": self.exchange_id,
                "entryReason": "synced_from_exchange",
                "decisionId": None,
            })
        wallet = num(usdt_account.get("usdtEquity")) or num(usdt_account.get("equity")) or 0
        available = num(usdt_account.get("available")) or num(usdt_account.get("crossMaxAvailable")) or 0
        return {
            "walletBalanceUsd": wallet,
            "equityUsd": wallet,
            "availableBalanceUsd": available,
            "unrealizedPnlUsd": unrealized_pnl,
            "openPositions": open_positions,
            "raw": {"account": usdt_account, "positions": positions_data},
        }

    def cancel_all_open_orders(self, config: dict[str, Any], symbol: str) -> Any:
        normalized = self.normalize_symbol(symbol)
        result: dict[str, Any] = {"cancelled": []}
        try:
            pending = self._signed_request_json(
                config, "GET", "/api/v2/mix/order/orders-pending",
                params={"symbol": normalized, "productType": _PRODUCT_TYPE},
            )
            order_list = pending.get("entrustedList") if isinstance(pending, dict) else (pending if isinstance(pending, list) else [])
            for row in (order_list or []):
                oid = str(row.get("orderId") or "").strip()
                if oid:
                    try:
                        self._signed_request_json(
                            config, "POST", "/api/v2/mix/order/cancel-order",
                            body={"symbol": normalized, "productType": _PRODUCT_TYPE, "orderId": oid},
                        )
                        result["cancelled"].append(oid)
                    except Exception:
                        pass
        except Exception:
            pass
        return result

    def place_market_order(
        self,
        config: dict[str, Any],
        *,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        side_upper = str(side or "").strip().upper()
        trade_side = "close" if reduce_only else "open"
        body: dict[str, Any] = {
            "symbol": normalized,
            "productType": _PRODUCT_TYPE,
            "marginMode": "crossed" if str(config.get("marginType") or "crossed").lower() != "isolated" else "isolated",
            "marginCoin": "USDT",
            "size": str(quantity),
            "side": side_upper,
            "tradeSide": trade_side,
            "orderType": "market",
        }
        result = self._signed_request_json(config, "POST", "/api/v2/mix/order/place-order", body=body)
        return result if isinstance(result, dict) else {}

    def place_protection_orders(
        self,
        config: dict[str, Any],
        *,
        symbol: str,
        position_side: str,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        if stop_loss is None and take_profit is None:
            return created
        normalized = self.normalize_symbol(symbol)
        close_side = "sell" if position_side == "long" else "buy"
        base: dict[str, Any] = {
            "symbol": normalized,
            "productType": _PRODUCT_TYPE,
            "marginCoin": "USDT",
            "side": close_side.upper(),
            "tradeSide": "close",
            "orderType": "market",
            "size": "0",
            "reduceOnly": "YES",
        }
        if stop_loss is not None:
            body = {**base, "triggerPrice": str(self.normalize_price(config, normalized, stop_loss)), "triggerType": "mark_price", "planType": "loss_plan"}
            try:
                r = self._signed_request_json(config, "POST", "/api/v2/mix/order/place-tpsl-order", body=body)
                if r:
                    created.append(r if isinstance(r, dict) else {"raw": r})
            except Exception:
                pass
        if take_profit is not None:
            body = {**base, "triggerPrice": str(self.normalize_price(config, normalized, take_profit)), "triggerType": "mark_price", "planType": "profit_plan"}
            try:
                r = self._signed_request_json(config, "POST", "/api/v2/mix/order/place-tpsl-order", body=body)
                if r:
                    created.append(r if isinstance(r, dict) else {"raw": r})
            except Exception:
                pass
        return created
