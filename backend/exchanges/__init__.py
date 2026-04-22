from __future__ import annotations

from typing import Any

from .base import ExchangeGateway


_GATEWAYS: dict[str, ExchangeGateway] = {}


def _ensure_gateways() -> None:
    if _GATEWAYS:
        return
    from .binance import BinanceGateway
    from .bitget import BitgetGateway
    from .bybit import BybitGateway
    from .okx import OkxGateway

    _GATEWAYS["binance"] = BinanceGateway()
    _GATEWAYS["bitget"] = BitgetGateway()
    _GATEWAYS["bybit"] = BybitGateway()
    _GATEWAYS["okx"] = OkxGateway()


def get_exchange_gateway(exchange_id: str | None) -> ExchangeGateway:
    _ensure_gateways()
    key = str(exchange_id or "binance").strip().lower() or "binance"
    if key not in _GATEWAYS:
        supported = ", ".join(sorted(_GATEWAYS))
        raise ValueError(f"Unsupported exchange: {key}. Supported exchanges: {supported}.")
    return _GATEWAYS[key]


def get_live_exchange_gateway(live_config: dict[str, Any] | None = None) -> ExchangeGateway:
    if live_config is None:
        from ..config import read_live_trading_config

        live_config = read_live_trading_config()
    return get_exchange_gateway(live_config.get("exchange"))


def get_active_exchange_gateway(
    exchange_id: str | None = None,
    settings: dict[str, Any] | None = None,
) -> ExchangeGateway:
    if exchange_id is not None:
        return get_exchange_gateway(exchange_id)
    if settings is None:
        from ..config import read_trading_settings

        settings = read_trading_settings()
    return get_exchange_gateway(settings.get("activeExchange"))


def active_exchange_id(
    exchange_id: str | None = None,
    settings: dict[str, Any] | None = None,
) -> str:
    if exchange_id is not None:
        return str(exchange_id).strip().lower() or "binance"
    if settings is None:
        from ..config import read_trading_settings

        settings = read_trading_settings()
    return str(settings.get("activeExchange") or "binance").strip().lower() or "binance"


def base_asset_for_symbol(symbol: str, exchange_id: str | None = None) -> str:
    normalized = str(symbol or "").strip().upper()
    try:
        return get_exchange_gateway(exchange_id).base_asset_from_symbol(normalized)
    except Exception:
        if normalized.endswith("USDT"):
            return normalized[:-4]
        if "-" in normalized:
            return normalized.split("-", 1)[0]
        return normalized
