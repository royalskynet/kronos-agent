# Changelog / 開發進程

All notable changes to this project, most recent first.

## Unreleased

### Session 1 (2026-04-17) — Core build + W0 validation + Lana philosophy alignment

**Scaffold**
- Initial freqtrade project structure with Bitget execution + Binance signal layer
- `LanaMMStrategy` with OI divergence, funding, taker, volume-structure signals
- Phase 0-3 configs, docker-compose stack, custom Dockerfile extending freqtrade
- Bilingual Traditional Chinese Telegram webhook (overrides native English)
- `scripts/daily_summary.py` + host cron at noon (qwen2.5:14b summarisation)

**Data pipeline**
- Binance `openInterestHist` / `takerlongshortRatio` / funding (no key)
- Bitget `history-fund-rate` / `account-long-short` (native execution venue)
- Bitget Wallet `copenapi.bgwapi.io` on-chain (holders, smart money, profit ratio)
- firecrawl self-host for Binance Square scraping
- camofox-browser for X anti-detection (supports cookie-based login)
- Redis `sentiment:{pair}` merged by 3 scraper workers

**Validation**
- W0 H1 hypothesis test: OI divergence edge
  - Small sample (5×14): Cohen's d = 0.268 (inflated by survivor bias)
  - Expanded sample (50×30): d = 0.061 (honest estimate)
  - Tightened (z_oi > 2.5 + chase-high): d = 0.098, MWU NOT significant → long-tail pattern confirmed
- Signal frequency analysis: initial defaults produced ~60 trades/week (too loose) → tightened to target 2-7/week

**Philosophy / 三層訊號重構**
- Layer 1 (Lana core): OI divergence mandatory AND
- Layer 2 (harness): funding / macro / volume mandatory AND
- Layer 3 (confirmation): sentiment / on-chain soft score
- OI removed from confirmation_score to prevent double-counting

**Privacy tooling**
- Removed hardcoded TG bot token / chat_id from all JSON configs and docker-compose
- Migrated to `${VAR}` env substitution + `.env` (gitignored)
- `scripts/setup_camofox.sh` for portable camofox build
- Full bilingual documentation suite (README, ARCHITECTURE, SETUP, STRATEGY, LESSONS)

**Integrations**
- Bitget official `agent_hub` MCP + 5 skills (macro-analyst, market-intel, news-briefing, sentiment-analyst, technical-analysis)
- Bitget Wallet AI Lab `bitget-wallet-mcp` as on-chain data backbone
- X cookies one-liner import (`scripts/import_x_cookies.sh`) with x.com/twitter.com domain filter

**qwen usage (local, free)**
- Per-post 3-class sentiment (bullish/neutral/bearish) in square_source
- Daily Chinese account summary
- Narrative agent default backend (P1+ opt-in)
- **Removed** entry/exit decision gate (too conservative, vetoed strong signals)

**Dry-run results (first hours)**
- TAO long 35min, tp_3R_trail, +1.03 USDT
- DOT long 50s, stoploss_on_exchange, -0.90 USDT
- Net +0.13 USDT, matches expected long-tail shape

**Reboot resilience**
- All 6 project containers `restart: unless-stopped`
- firecrawl override added for auto-restart
- Ollama as macOS login item
- X cookies persist in docker volume

### Artifacts / 資產

| File | Purpose |
|---|---|
| `user_data/strategies/lana_mm_strategy.py` | Main 3-layer strategy |
| `user_data/signals/*.py` | Individual signal calculators |
| `user_data/data_providers/external_data.py` | Binance + Bitget signal fetcher with TTL cache |
| `user_data/data_providers/square_source.py` | Binance Square scraper + qwen sentiment |
| `user_data/data_providers/x_source.py` | X scraper via camofox |
| `user_data/data_providers/onchain_bgw.py` | Bitget Wallet copenapi client |
| `user_data/data_providers/onchain_source.py` | On-chain worker |
| `user_data/agents/narrative.py` | qwen/Anthropic narrative (P1+) |
| `user_data/notebooks/W0_full_results.md` | Full W0 validation report |
| `scripts/w0_validate.py` | Standalone W0 H1 runner with CLI args |
| `scripts/signal_frequency_check.py` | Simulated trade count from cached parquet |
| `scripts/daily_summary.py` | 12:00 cron report |
| `scripts/setup_camofox.sh` | Portable camofox bootstrap |
| `scripts/import_x_cookies.sh` | X cookies import one-liner |

### Known open items / 已知待辦

- Broader BTC dominance informative_pair filter (currently per-pair EMA(48) proxy)
- Custom PairList `SquareHotnessPairList` waiting on freqtrade schema workaround
- Bot-vs-human classifier training data (sklearn) not yet collected
- TradingAgents LangGraph integration for P2+
- Grafana dashboards for live P&L / signal attribution

### Session token/cost summary / 本次 session 成本

- Main Claude session: conversation / coding
- Sub-agents launched: 3 (W0 validation ×2, signal frequency check)
- qwen calls: via local Ollama, zero API cost
- Anthropic API direct: 0 calls so far (P0 disabled LLM gating)
- TG messages sent: ~15 push notifications to user chat

## 2026-04-16 and earlier

Not applicable — project started 2026-04-17.
