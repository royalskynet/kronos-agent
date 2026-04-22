from __future__ import annotations

from copy import deepcopy
from typing import Any


EXCHANGE_CATALOG: dict[str, dict[str, Any]] = {
    "binance": {
        "id": "binance",
        "label": "Binance",
        "implemented": True,
        "marketSupported": True,
        "tradingSupported": True,
        "market": "usdm_futures",
        "defaultBaseUrl": "https://fapi.binance.com",
        "apiKeyPlaceholder": "Binance API key",
        "apiSecretPlaceholder": "Binance API secret",
        "defaultLeverageMax": 125,
        "notes": "当前版本已实现 Binance USDT 永续的行情与实盘接口。",
    },
    "okx": {
        "id": "okx",
        "label": "OKX",
        "implemented": True,
        "marketSupported": True,
        "tradingSupported": True,
        "market": "swap",
        "defaultBaseUrl": "https://www.okx.com",
        "apiKeyPlaceholder": "OKX API key",
        "apiSecretPlaceholder": "OKX API secret",
        "apiPassphrasePlaceholder": "OKX API passphrase",
        "requiresPassphrase": True,
        "defaultLeverageMax": 125,
        "notes": "已实现 OKX 永续合约的行情与实盘接口，需要填写 API Passphrase。",
    },
    "bybit": {
        "id": "bybit",
        "label": "Bybit",
        "implemented": True,
        "marketSupported": True,
        "tradingSupported": False,
        "market": "linear",
        "defaultBaseUrl": "https://api.bybit.com",
        "apiKeyPlaceholder": "Bybit API key",
        "apiSecretPlaceholder": "Bybit API secret",
        "apiPassphrasePlaceholder": "",
        "requiresPassphrase": False,
        "defaultLeverageMax": 125,
        "notes": "已实现 Bybit 线性永续公共行情，实盘接口会在后续版本接入。",
    },
    "gateio": {
        "id": "gateio",
        "label": "Gate.io",
        "implemented": False,
        "marketSupported": False,
        "tradingSupported": False,
        "market": "futures",
        "defaultBaseUrl": "https://api.gateio.ws",
        "apiKeyPlaceholder": "Gate.io API key",
        "apiSecretPlaceholder": "Gate.io API secret",
        "apiPassphrasePlaceholder": "",
        "requiresPassphrase": False,
        "defaultLeverageMax": 100,
        "notes": "即将支持 Gate.io 合约接口。",
    },
    "bitget": {
        "id": "bitget",
        "label": "Bitget",
        "implemented": True,
        "marketSupported": True,
        "tradingSupported": True,
        "market": "usdt_futures",
        "defaultBaseUrl": "https://api.bitget.com",
        "apiKeyPlaceholder": "Bitget API key",
        "apiSecretPlaceholder": "Bitget API secret",
        "apiPassphrasePlaceholder": "Bitget API passphrase",
        "requiresPassphrase": True,
        "defaultLeverageMax": 125,
        "notes": "Bitget USDT-M 永续合约行情与实盘接口。需填写 API Passphrase。",
    },
}

EXCHANGE_CATALOG["binance"]["apiPassphrasePlaceholder"] = ""
EXCHANGE_CATALOG["binance"]["requiresPassphrase"] = False

DEFAULT_EXCHANGE_ID = "binance"


def exchange_catalog_payload() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in EXCHANGE_CATALOG.values()]


def exchange_config(exchange_id: str | None = None) -> dict[str, Any]:
    normalized = normalize_exchange_id(exchange_id)
    return deepcopy(EXCHANGE_CATALOG[normalized])


def exchange_supports(exchange_id: Any, capability: str = "market") -> bool:
    key = str(exchange_id or DEFAULT_EXCHANGE_ID).strip().lower() or DEFAULT_EXCHANGE_ID
    meta = EXCHANGE_CATALOG.get(key)
    if not meta:
        return False
    if capability == "trade":
        return meta.get("tradingSupported") is True
    if capability == "market":
        return meta.get("marketSupported") is True
    return meta.get("implemented") is True


def normalize_exchange_id(
    exchange_id: Any,
    *,
    implemented_only: bool = False,
    capability: str | None = None,
) -> str:
    key = str(exchange_id or DEFAULT_EXCHANGE_ID).strip().lower() or DEFAULT_EXCHANGE_ID
    if key not in EXCHANGE_CATALOG:
        return DEFAULT_EXCHANGE_ID
    if capability and not exchange_supports(key, capability):
        return DEFAULT_EXCHANGE_ID
    if implemented_only and not EXCHANGE_CATALOG[key]["implemented"]:
        return DEFAULT_EXCHANGE_ID
    return key
