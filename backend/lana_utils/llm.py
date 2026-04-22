"""OpenRouter dynamic free-model router client.

使用 vendored model_router (royalskynet/openrouter-dynamic-free-model-router),
動態從 OpenRouter API 挑最佳免費模型 (按 context_length 排序, 排除非文字/限流)
並自動 fallback 到下一個模型 / 本地 Ollama. 零依賴 stdlib.

env:
  OPENROUTER_API_KEY  必填 (model_router 也讀這個)
  OPENROUTER_BASE     預設 https://openrouter.ai/api/v1
  OPENROUTER_REFERER  HTTP-Referer header
  OPENROUTER_TITLE    X-Title header
  LLM_NEED_TOOLS      "1" 時挑支援 tool calling 的模型
  LLM_MODEL_CACHE_SEC 選好的 model id 快取幾秒 (預設 300)
  LLM_MAX_FALLBACKS   單次請求最多切幾個模型 (預設 3)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from user_data.utils import model_router

logger = logging.getLogger(__name__)

BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
KEY = os.getenv("OPENROUTER_API_KEY", "")
REFERER = os.getenv("OPENROUTER_REFERER", "https://github.com/royalskynet/binance-trading-agent")
TITLE = os.getenv("OPENROUTER_TITLE", "Lana Trading Agent")

NEED_TOOLS = os.getenv("LLM_NEED_TOOLS", "0") == "1"
MODEL_CACHE_SEC = int(os.getenv("LLM_MODEL_CACHE_SEC", "300"))
MAX_FALLBACKS = int(os.getenv("LLM_MAX_FALLBACKS", "3"))

_cached: dict[str, Any] = {"id": None, "ts": 0.0}


def _strip_prefix(router_id: str) -> str:
    """router 回 'openrouter/xxx:free' → OpenRouter API 要 'xxx:free'."""
    if router_id.startswith("openrouter/"):
        return router_id[len("openrouter/"):]
    return router_id


def _pick(exclude: set[str] | None = None) -> str:
    """挑模型, 回 OpenRouter model id (無 openrouter/ 前綴)."""
    now = time.time()
    if not exclude and _cached["id"] and (now - _cached["ts"]) < MODEL_CACHE_SEC:
        return _cached["id"]
    picked = model_router.pick_best(need_tools=NEED_TOOLS, exclude=exclude or set())
    rid = picked.get("id")
    if not rid:
        raise RuntimeError(f"model_router found nothing: {picked.get('error')}")
    if picked.get("fallback"):
        raise RuntimeError(f"OpenRouter 無 free 模型可用, router 要 fallback 到本地 ({rid})")
    return _strip_prefix(rid)


def _cache_success(mid: str) -> None:
    """成功呼叫後, 才把該模型寫入 cache (避免 cache 到限流模型)."""
    _cached["id"] = mid
    _cached["ts"] = time.time()


def chat(
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = 0.3,
    timeout: float = 30.0,
    max_tokens: int | None = None,
) -> str:
    """POST OpenRouter chat completions, 回 content 字串.

    自動 fallback: 遇 4xx/5xx/超時 時問 model_router 要下一個模型, 最多 MAX_FALLBACKS 次.
    全數失敗 raise.
    """
    if not KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    tried: set[str] = set()
    last_err: Exception | None = None

    for attempt in range(MAX_FALLBACKS + 1):
        exclude = {f"openrouter/{m}" for m in tried} if tried else None
        try:
            mid = _pick(exclude=exclude)
        except Exception as e:
            raise RuntimeError(f"No model available (tried {tried}): {e}") from e

        body: dict[str, Any] = {
            "model": mid,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body["max_tokens"] = max_tokens

        try:
            r = httpx.post(
                f"{BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {KEY}",
                    "HTTP-Referer": REFERER,
                    "X-Title": TITLE,
                },
                json=body,
                timeout=timeout,
            )
            r.raise_for_status()
            payload = r.json()
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError(f"OpenRouter empty response: {payload}")
            content = choices[0]["message"]["content"] or ""
            _cache_success(mid)
            if attempt > 0:
                logger.info("llm: fell back to %s after %d tries", mid, attempt)
            return content
        except Exception as e:
            logger.warning("llm: model %s failed (%s), trying next", mid, e)
            tried.add(mid)
            _cached["id"] = None  # invalidate cache
            last_err = e
            continue

    raise RuntimeError(f"All {MAX_FALLBACKS + 1} models failed, last err: {last_err}")
