"""NarrativeAgent — 從廣場/X 原文解析敘事. 預設 OpenRouter 免費模型.

env 控制後端:
  NARRATIVE_BACKEND=openrouter  (預設, nemotron/qwen/llama free 輪替)
  NARRATIVE_BACKEND=anthropic   (Haiku 4.5, P2+ 升級用)

硬規則: 輸入嚴格只有 public market content. 不看 user 推文/倉位/歷史決策,
杜絕「幣安人生」類 LLM 聯想錯誤.
"""
from __future__ import annotations

import json
import logging
import os

import redis

from user_data.utils.llm import chat as llm_chat

logger = logging.getLogger(__name__)

BACKEND = os.getenv("NARRATIVE_BACKEND", "openrouter")
ANTHROPIC_MODEL = os.getenv("NARRATIVE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """你是 crypto narrative analyzer. 給你某 ticker 的廣場/X 帖子, 輸出 JSON:
{"narrative": "<1 句中文>", "confidence": 0.0-1.0, "is_bot_inflated": true|false, "drivers": ["<關鍵字>", ...]}

規則:
- 只根據提供的帖子, 不能用你對該幣的外部知識
- confidence 是「敘事是否一致」不是「是否看多」
- is_bot_inflated = true 若 >60% 帖看起來模板化
- 只回 JSON, 不加其他文字"""


class NarrativeAgent:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )

    def _openrouter(self, user_msg: str) -> dict:
        raw = llm_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            json_mode=True,
            temperature=0.4,
            timeout=30.0,
        )
        return json.loads(raw or "{}")

    def _anthropic(self, user_msg: str) -> dict:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        return json.loads(text)

    def analyze(self, pair: str, ticker: str, posts: list[str], x_posts: list[str]) -> dict:
        user_msg = json.dumps(
            {"ticker": ticker, "square_posts": posts[:15], "x_posts": x_posts[:15]},
            ensure_ascii=False,
        )
        try:
            if BACKEND == "anthropic":
                parsed = self._anthropic(user_msg)
            else:
                parsed = self._openrouter(user_msg)
        except Exception as e:
            logger.warning("narrative agent err (%s): %s", BACKEND, e)
            parsed = {"narrative": "", "confidence": 0.0, "is_bot_inflated": True, "drivers": []}

        key = f"sentiment:{pair.replace('/', '_').replace(':', '')}"
        existing = self.redis.get(key)
        feats = json.loads(existing) if existing else {}
        feats["narrative"] = parsed.get("narrative", "")
        feats["narrative_confidence"] = float(parsed.get("confidence", 0.0))
        feats["narrative_bot_inflated"] = bool(parsed.get("is_bot_inflated", False))
        feats["narrative_drivers"] = parsed.get("drivers", [])
        feats["narrative_backend"] = BACKEND
        self.redis.setex(key, 600, json.dumps(feats))
        return parsed
