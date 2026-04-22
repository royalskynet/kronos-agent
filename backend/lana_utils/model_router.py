#!/usr/bin/env python3
"""Dynamic free model router — standalone, zero dependencies.

Discovers available LLM backends (OpenRouter, Ollama, LM Studio, vLLM, LocalAI,
or any OpenAI-compatible endpoint), selects the best free model by capability,
and provides automatic fallback.

No external dependencies — uses Python stdlib only.

Usage:
    python3 model_router.py env                                      # Detect environment
    python3 model_router.py list [--tools] [--top N] [--local]       # List models
    python3 model_router.py pick [--tools] [--exclude M1,M2]         # Pick best
    python3 model_router.py fallback --failed M [--tried M1,M2] [--tools]
    python3 model_router.py local                                    # List local models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Non-text models to exclude (music/image/aggregator)
# ---------------------------------------------------------------------------
_EXCLUDED_PATTERNS = frozenset({
    "lyria", "musicgen", "imagen", "dall-e", "stable-diffusion",
    "elephant-alpha", "elephant-beta",
})

# ---------------------------------------------------------------------------
# Known local LLM endpoints to probe
# ---------------------------------------------------------------------------
_DEFAULT_LOCAL_ENDPOINTS = [
    {"name": "Ollama",    "url": "http://127.0.0.1:11434", "api": "/api/tags",  "prefix": "ollama_chat"},
    {"name": "LM Studio", "url": "http://127.0.0.1:1234",  "api": "/v1/models", "prefix": "openai"},
    {"name": "vLLM",      "url": "http://127.0.0.1:8000",  "api": "/v1/models", "prefix": "openai"},
    {"name": "LocalAI",   "url": "http://127.0.0.1:8080",  "api": "/v1/models", "prefix": "openai"},
]


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def detect_environment() -> dict:
    """Probe the system and report what LLM backends are available.

    Returns a dict describing:
      - openrouter: whether OPENROUTER_API_KEY is set and the API is reachable
      - local_backends: list of detected local LLM servers with their models
      - summary: human-readable summary
    """
    result = {
        "openrouter": {"available": False, "api_key_set": False, "reachable": False},
        "local_backends": [],
        "custom_endpoints": [],
        "summary": [],
    }

    # Check OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    result["openrouter"]["api_key_set"] = bool(api_key)
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            req = Request("https://openrouter.ai/api/v1/models", headers=headers)
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    result["openrouter"]["reachable"] = True
                    result["openrouter"]["available"] = True
                    result["summary"].append("OpenRouter: available (API key set)")
        except (URLError, OSError):
            result["summary"].append("OpenRouter: API key set but unreachable")
    else:
        # Try without auth
        try:
            req = Request("https://openrouter.ai/api/v1/models")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    result["openrouter"]["reachable"] = True
                    result["openrouter"]["available"] = True
                    result["summary"].append("OpenRouter: available (no API key, may be rate limited)")
        except (URLError, OSError):
            result["summary"].append("OpenRouter: not reachable")

    # Check local backends
    for ep in _get_local_endpoints():
        url = ep["url"].rstrip("/") + ep["api"]
        try:
            req = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            models = _parse_local_response(ep, data)
            if models:
                backend = {
                    "name": ep["name"],
                    "url": ep["url"],
                    "model_count": len(models),
                    "models": [m["name"] for m in models],
                }
                result["local_backends"].append(backend)
                result["summary"].append(
                    f"{ep['name']} ({ep['url']}): {len(models)} model(s) — {', '.join(m['name'] for m in models[:3])}"
                )
        except (URLError, OSError, json.JSONDecodeError):
            continue

    if not result["openrouter"]["available"] and not result["local_backends"]:
        result["summary"].append("WARNING: No LLM backends detected. Set OPENROUTER_API_KEY or start a local server.")

    return result


# ---------------------------------------------------------------------------
# Local model detection
# ---------------------------------------------------------------------------
def _get_local_endpoints() -> list[dict]:
    """Build list of local endpoints to probe, including env-var overrides."""
    endpoints = list(_DEFAULT_LOCAL_ENDPOINTS)

    # Allow user to add custom OpenAI-compatible endpoints
    custom = os.environ.get("MODEL_ROUTER_LOCAL_ENDPOINTS", "").strip()
    if custom:
        for entry in custom.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split("=", 1)
            name = parts[0].strip() if len(parts) > 1 else "Custom"
            url = parts[-1].strip()
            endpoints.append({
                "name": name,
                "url": url,
                "api": "/v1/models",
                "prefix": "openai",
            })

    # Env var overrides for known backends
    for ep in endpoints:
        env_name = ep["name"].upper().replace(" ", "_") + "_BASE_URL"
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            ep["url"] = env_val

    return endpoints


def _parse_local_response(ep: dict, data: dict) -> list[dict]:
    """Parse model list from a local backend response."""
    models = []
    if ep["name"] == "Ollama":
        for m in data.get("models", []):
            name = m.get("name", "")
            if not name:
                continue
            size_bytes = m.get("size", 0)
            models.append({
                "id": f"{ep['prefix']}/{name}",
                "name": name,
                "provider": ep["name"],
                "base_url": ep["url"],
                "size_gb": round(size_bytes / (1024 ** 3), 1) if size_bytes else None,
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
                "family": m.get("details", {}).get("family", ""),
            })
    else:
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid:
                continue
            models.append({
                "id": f"{ep['prefix']}/{mid}",
                "name": mid,
                "provider": ep["name"],
                "base_url": ep["url"],
                "size_gb": None,
                "parameter_size": "",
                "family": "",
            })
    return models


def detect_local_models() -> list[dict]:
    """Probe all local endpoints and return available models."""
    results = []
    for ep in _get_local_endpoints():
        url = ep["url"].rstrip("/") + ep["api"]
        try:
            req = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            results.extend(_parse_local_response(ep, data))
        except (URLError, OSError, json.JSONDecodeError):
            continue
    return results


# ---------------------------------------------------------------------------
# OpenRouter free model discovery
# ---------------------------------------------------------------------------
def fetch_free_models(need_tools: bool = False) -> list[dict]:
    """Query OpenRouter API for available free models.

    Returns list of dicts sorted by context_length descending.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request("https://openrouter.ai/api/v1/models", headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (URLError, OSError) as exc:
        print(json.dumps({"error": f"Failed to fetch OpenRouter models: {exc}"}), file=sys.stderr)
        return []

    candidates = []
    for m in data.get("data", []):
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "1") or "1")
        completion_price = float(pricing.get("completion", "1") or "1")
        if prompt_price != 0 or completion_price != 0:
            continue

        model_id = m.get("id", "")
        if not model_id or model_id == "openrouter/free":
            continue

        model_lower = model_id.lower()
        if any(pat in model_lower for pat in _EXCLUDED_PATTERNS):
            continue

        if need_tools:
            supported = m.get("supported_parameters", [])
            if "tools" not in supported:
                continue

        candidates.append({
            "id": f"openrouter/{model_id}",
            "context_length": m.get("context_length", 0),
            "name": m.get("name", model_id),
            "supported_parameters": m.get("supported_parameters", []),
        })

    candidates.sort(key=lambda x: x["context_length"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def pick_best(need_tools: bool = False, exclude: set[str] | None = None) -> dict:
    """Pick the best available free model, falling back to local if needed."""
    exclude = exclude or set()
    models = fetch_free_models(need_tools=need_tools)
    for m in models:
        if m["id"] not in exclude:
            return m
    # Fallback to local
    for local in detect_local_models():
        if local["id"] not in exclude:
            return {**local, "fallback": True}
    return {"id": None, "error": "No free or local models available"}


def pick_fallback(failed: str, tried: set[str], need_tools: bool = False) -> dict:
    """Pick next model after failure, skipping already-tried models."""
    tried.add(failed)
    models = fetch_free_models(need_tools=need_tools)
    for m in models:
        if m["id"] not in tried:
            return m
    for local in detect_local_models():
        if local["id"] not in tried:
            return {**local, "fallback": True}
    return {"id": None, "error": "All models exhausted", "tried": list(tried)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dynamic free model router — discover, select, and fallback across OpenRouter and local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # env
    sub.add_parser("env", help="Detect available LLM backends and report environment status")

    # list
    list_cmd = sub.add_parser("list", help="List available free models")
    list_cmd.add_argument("--tools", action="store_true", help="Only models supporting tool calling")
    list_cmd.add_argument("--top", type=int, default=10, help="Number of models to show (default: 10)")
    list_cmd.add_argument("--local", action="store_true", help="Include local models in the list")

    # pick
    pick_cmd = sub.add_parser("pick", help="Pick the best available model")
    pick_cmd.add_argument("--tools", action="store_true", help="Require tool calling support")
    pick_cmd.add_argument("--exclude", type=str, default="", help="Comma-separated model IDs to exclude")

    # fallback
    fb_cmd = sub.add_parser("fallback", help="Pick next model after failure")
    fb_cmd.add_argument("--failed", required=True, help="Model that just failed")
    fb_cmd.add_argument("--tried", type=str, default="", help="Comma-separated already-tried model IDs")
    fb_cmd.add_argument("--tools", action="store_true", help="Require tool calling support")

    # local
    sub.add_parser("local", help="Detect and list locally running models")

    args = parser.parse_args()

    if args.command == "env":
        result = detect_environment()
        print(json.dumps(result, indent=2))

    elif args.command == "list":
        models = fetch_free_models(need_tools=args.tools)
        result = models[:args.top]
        output = {"openrouter_models": result, "openrouter_total": len(models)}
        if args.local:
            local = detect_local_models()
            output["local_models"] = local
            output["local_total"] = len(local)
        print(json.dumps(output, indent=2))

    elif args.command == "pick":
        exclude = set(filter(None, args.exclude.split(",")))
        result = pick_best(need_tools=args.tools, exclude=exclude)
        print(json.dumps(result, indent=2))

    elif args.command == "fallback":
        tried = set(filter(None, args.tried.split(",")))
        result = pick_fallback(args.failed, tried, need_tools=args.tools)
        print(json.dumps(result, indent=2))

    elif args.command == "local":
        models = detect_local_models()
        if not models:
            print(json.dumps({
                "models": [],
                "total": 0,
                "hint": "No local LLM servers detected. Checked: "
                        + ", ".join(f"{ep['name']} ({ep['url']})" for ep in _get_local_endpoints())
                        + ". Start a server or set MODEL_ROUTER_LOCAL_ENDPOINTS=name=http://host:port"
            }, indent=2))
        else:
            print(json.dumps({"models": models, "total": len(models)}, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
