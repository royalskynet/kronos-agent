# Binance 合約交易 Agent — 膠水版設計 (100U → 100K)

## Context

- **資金曲線**：100U → 100,000U（1000x）。
- **靈感**：Lana (`@lanaaielsa`) 抓 MM 買壳 + Binance 廣場情緒 + OI/價格背離。
- **原帖 bug**：Agent 選「幣安人生」因 user 推文被 CZ 轉 → **關注點未分離**，LLM 聯想錯誤。
- **Lana 實盤**（user 提供）：6 筆全虧、75x 槓桿、固定 $200-250 $ 止損、靠 10-50R 巨勝者拉均值。
- **100U 硬限制**：Binance alt 最低名義 5U、BTC 100U、ETH 20U；Bitget alt 5U。
- **雙層架構**：Binance public REST 只讀作信號源（廣場/OI/funding 免 key），**執行切 Bitget**（user 無法用 Binance 子帳戶）。Bitget freqtrade 原生支援 isolated futures + stoploss_on_exchange + passphrase。
- **Bitget 限制**：自 2026-02-11 起新註冊主帳戶的子帳戶合約 5x 上限 → 用主帳戶 API key 可達 150x；本專案 P0 20x 遠低於硬頂。
- `openInterestHist` 5m 粒度、30 天回溯、1000 req/5min。

**核心原則**：**能抄不寫、能連不造**。不從零搭框架，fork 成熟項目做膠水。

---

## 膠水選型（先找成熟方案）

| 層 | 選用 | 為何 | 替代 |
|---|------|------|-----|
| **策略框架 / 執行 / 回測** | **freqtrade** (39.9k★, MIT, CCXT) | 成熟生態、FreqAI ML、Bitget + Binance 現成、backtest + live + hyperopt + dry-run 全套 | Jesse（live 收費）、Nautilus（太重）、OctoBot |
| **執行交易所** | **Bitget** USDT-M 永續 | freqtrade 原生支援 isolated mode、stoploss_on_exchange、passphrase、150x 主帳戶上限 | — |
| **信號交易所** | **Binance** public REST | 流動性最深、廣場/OI/funding 免 key、alpha 最強 | Bybit（若 Binance 公開端點被封） |
| **LLM 多 agent 層** | **TradingAgents** (TauricResearch, LangGraph) | 已有 Bull/Bear/Risk/News/Sentiment agent 模板、支援 Claude/Ollama | 自建 LangGraph |
| **交易所 API** | **CCXT**（freqtrade 已封裝） | — | — |
| **技術指標** | **TA-Lib / pandas-ta** | 100+ 指標、freqtrade 原生支持 | — |
| **OI 背離** | **自寫**（無成熟 OI 專用庫）✱ | Lana 核心 edge | 參考 `korfer/Crypto-Divergence-Detector` 思路 |
| **廣場爬取** | **firecrawl (self-host, user 已有)** + camofox 退路 | Next.js SSR 公開頁、firecrawl 原生 markdown + change-tracking 偵測新帖、P95 3.4s；被風控則切 camofox | Bright Data / ScrapingBee |
| **X/Twitter 情緒** | **camofox-browser (user 已有)** + cookie 認證 | Camoufox C++ 級指紋偽造、內建 `@twitter_search` macro、支援 cookie import 做登入態、session 隔離；X 嚴格反爬 firecrawl 不適用 | xint 已移除不可復用 |
| **NLP 情緒分數** | **VADER** (初期) → FinBERT / crypto-BERT (P2+) | 輕量快速 | transformers |
| **Bot 分類器** | **sklearn LR + xgboost** | 輕量、六維特徵訓得動 | — |
| **Orchestration** | **LangGraph** (TradingAgents 自帶) | 與 TradingAgents 一致 | — |
| **任務排程** | **freqtrade 內建 scheduler** + **cron** | 無需自建 | — |
| **存儲** | **Postgres + Redis** (docker-compose) | freqtrade 原生支持 | sqlite (freqtrade default) |
| **監控** | **freqtrade Telegram bot + Grafana** | freqtrade 內建 TG | 已有 Hermes TG gateway |
| **告警** | **Hermes (nia profile)** | user 現有基建 | — |
| **覆盤 / 覆盤歸因** | **freqtrade `backtesting-analysis` + `plot-dataframe`** + 自寫週報 | freqtrade 自帶 | — |

✱ OI 背離無成熟庫：freqtrade 內寫成 `Hyperoptable` indicator。

---

## 膠水架構（用既有組件組裝）

```
┌─────────────────────────────────────────────────────────────┐
│ DATA SOURCES                                                │
│  ├─ freqtrade 內建 Binance data provider (K/OI/funding)      │
│  ├─ SquareSource → firecrawl scrape+change-tracking          │
│  │    (廣場新帖偵測、markdown 清洗) → pairlist 熱度            │
│  ├─ XSource → camofox-browser `@twitter_search` + cookies    │
│  │    (CZ/He Yi/KOL 轉發監聽 + 新幣 ticker 搜尋)               │
│  └─ 結果寫 Redis → freqtrade custom PairList 讀                │
└─────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ FREQTRADE CORE (既有框架，不動)                              │
│  ├─ Pairlist: VolumePairList + 自寫 SquareHotnessPairList    │
│  ├─ Strategy: class LanaMMStrategy(IStrategy):               │
│  │    populate_indicators: OI背離 + funding + taker + 情緒    │
│  │    populate_entry_trend / exit_trend                      │
│  │    custom_stoploss, custom_exit (time stop)               │
│  │    protections: CooldownPeriod, StoplossGuard, LowProfit  │
│  ├─ Protections (freqtrade 內建熔斷)                          │
│  ├─ Dry-run + Backtest + Hyperopt + Live                     │
│  └─ Telegram bot (freqtrade 自帶)                             │
└─────────────────────────────────────────────────────────────┘
                        ▲ (P2+ 注入)
┌─────────────────────────────────────────────────────────────┐
│ TRADINGAGENTS (LLM 層，plug-in)                               │
│  ├─ Narrative Agent (Haiku 4.5)                              │
│  ├─ Bull/Bear Debate (Sonnet 4.6)                            │
│  ├─ Risk LLM (Opus 4.7)                                      │
│  └─ 輸出 → freqtrade strategy 當 extra feature column         │
└─────────────────────────────────────────────────────────────┘
```

**選擇 freqtrade 而非從零搭的理由**：
- 倉位 / stop / trailing / time-stop / 熔斷 / dry-run / backtest / hyperopt / TG bot **全部內建**。
- 我們只需要寫 **1 個 Strategy 類 + 1 個 PairList plugin + 2 個自訂信號** 就能跑起來。省 6-8 週。
- user 已有 Hermes (nia/alice/koko/wellally) + MCP 生態，freqtrade 的 TG + Webhook 能直接接 Hermes。

---

## Phase 階梯（資金門檻硬規則）

| Phase | 本金 | 槓桿 | 單筆風險 | stop% | 並發 | 允許標的 | LLM | freqtrade 模式 |
|-------|------|-----|---------|------|-----|---------|-----|--------------|
| **P0** | 100-500U | 20x | 1% ($1-5) | 2-5% | 1 | alt-only (跳過 BTC/ETH) | 無 | live + protections |
| **P1** | 500-2K | 20-50x | 1% | 1.5-3% | 2 | alt + 中流動性 | Haiku narrative (freqtrade data provider 注入 LLM column) | live |
| **P2** | 2K-10K | 50-75x | 1-1.5% | 1-2% | 3-5 | 全 alt | TradingAgents 全棧 | live |
| **P3** | 10K-100K | 75-100x | 1-2% | 0.8-1.5% | 5-8 | 加自上線檢測 | 全 + 自動覆盤 agent | live + multi-bot |

**升級條件**（同時滿足）：淨值達下階 2x、30 天勝率 >35% 或 PF>1.3、最大回撤 <30%。

**熔斷**（走 freqtrade Protections）：
- `StoplossGuard`: 連 3 敗 stop 24h
- `MaxDrawdown`: 日內 >5% stop 24h
- `CooldownPeriod`: 單標 2 次虧損 → 30 天黑名單
- `LowProfitPairs`: 30 天 PF<1 自動下架

---

## Repo 結構（freqtrade 項目規範）

```
binance-trading-agent/
├── DESIGN.md                          # 本文
├── README.md
├── docker-compose.yml                 # freqtrade + postgres + redis + grafana
├── user_data/                         # freqtrade 標準佈局
│   ├── strategies/
│   │   └── lana_mm_strategy.py        # 主策略 (populate_indicators / entry / exit)
│   ├── pairlists/
│   │   └── square_hotness.py          # 自訂 PairList plugin
│   ├── hyperopts/
│   │   └── lana_hyperopt.py
│   ├── freqaimodels/                  # 若 P2+ 用 FreqAI ML
│   ├── data_providers/
│   │   ├── square_source.py           # firecrawl client (scrape + change-tracking)
│   │   ├── x_source.py                # camofox REST client (twitter_search + cookies)
│   │   ├── oi_stats.py                # openInterestHist 封裝
│   │   └── news_adapter.py            # user 現有新聞源接入
│   ├── signals/
│   │   ├── oi_divergence.py           # 核心 edge
│   │   ├── funding_anomaly.py
│   │   ├── taker_imbalance.py
│   │   ├── volume_structure.py
│   │   └── zscore.py
│   ├── bot_filter/
│   │   ├── features.py
│   │   └── classifier.pkl             # 訓練產出
│   ├── agents/                        # P1+ 啟用
│   │   ├── narrative.py               # 調 Haiku
│   │   ├── tradingagents_bridge.py    # 橋接 TauricResearch/TradingAgents
│   │   └── risk_sanity.py
│   ├── configs/
│   │   ├── config_phase0.json         # freqtrade config
│   │   ├── config_phase1.json
│   │   ├── config_phase2.json
│   │   └── config_phase3.json
│   ├── notebooks/
│   │   ├── W0_assumption_validation.ipynb  # ⭐ 必做在寫碼前
│   │   └── W2_signal_calibration.ipynb
│   └── backtest_results/
├── vendored/
│   └── TradingAgents/                 # git submodule, TauricResearch/TradingAgents
├── scripts/
│   ├── setup.sh                       # docker-compose up + freqtrade download-data
│   ├── run_dry.sh
│   └── run_live_p0.sh
└── tests/
    ├── test_signals.py
    └── test_pairlist.py
```

---

## W0 假設驗證（⭐ 寫碼前必做）

**用 freqtrade `download-data` 就能拉 6 個月 K 線 + funding，不用自己寫爬蟲。**

```bash
freqtrade download-data --exchange binance --trading-mode futures \
  --timeframes 5m 15m 1h --days 180 --pairs-file p0_universe.json
```

OI 歷史用 `openInterestHist` 直接 curl + pandas 存 parquet。

**H1**: OI 48h +2σ & 價格 <1σ → 未來 24-72h 超額收益 t-test p<0.05？
**H2**: 廣場發帖 +3σ & 獨立作者 > N → 未來收益？
**H3**: H1 ∩ H2 組合是否顯著？

**無顯著性就 pivot。** 寫 agent 前先驗信號有 edge。

---

## 實施路線圖

| 週 | 內容 | 用到的既有輪子 |
|----|------|-------------|
| **W0** | 假設驗證 notebook | freqtrade download-data, pandas, scipy |
| **W1** | fork freqtrade 模板、寫 `LanaMMStrategy` 骨架 + OI/funding 信號 | freqtrade + TA-Lib |
| **W2** | SquareSource (firecrawl) + XSource (camofox + cookies) + Bot classifier + 合成情緒 column | firecrawl self-host + camofox-browser + sklearn + VADER |
| **W3** | dry-run 7 天 + hyperopt 權重 | freqtrade hyperopt |
| **W4** | **P0 實盤 100U**、freqtrade TG bot 接 Hermes nia | freqtrade TG |
| **W5** | 加 Narrative Agent (Haiku 4.5) → P1 | 自寫 (user 已有 Anthropic SDK) |
| **W6** | 接入 TradingAgents submodule、Bull/Bear debate → P2 | TauricResearch/TradingAgents |
| **W7+** | FreqAI ML 層（可選）、自動週報、P3 規模化 | FreqAI + Grafana |

---

## Phase 0 實操（100U config 範例）

```json
// configs/config_phase0.json (freqtrade config 片段)
{
  "max_open_trades": 1,
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.95,
  "dry_run": false,
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "exchange": {
    "name": "binance",
    "pair_whitelist": [],                  // 由 SquareHotnessPairList 動態填
    "pair_blacklist": ["BTC/USDT:USDT", "ETH/USDT:USDT"]
  },
  "pairlists": [
    {"method": "VolumePairList", "number_assets": 50, "sort_key": "quoteVolume"},
    {"method": "SquareHotnessPairList", "number_assets": 10},
    {"method": "ShuffleFilter"}
  ],
  "protections": [
    {"method": "StoplossGuard", "lookback_period_candles": 24, "trade_limit": 3, "stop_duration_candles": 288, "only_per_pair": false},
    {"method": "CooldownPeriod", "stop_duration_candles": 12},
    {"method": "MaxDrawdown", "lookback_period_candles": 96, "trade_limit": 5, "stop_duration_candles": 288, "max_allowed_drawdown": 0.05},
    {"method": "LowProfitPairs", "lookback_period_candles": 2880, "trade_limit": 2, "stop_duration_candles": 10080, "required_profit": 0.0}
  ],
  "telegram": {"enabled": true, "token": "...", "chat_id": "..."},
  "api_server": {"enabled": true}
}
```

**strategy 關鍵點**：
- `custom_stake_amount`：按 equity × 1% / stop% 反算，下限貼 $5 min notional
- `custom_stoploss`：trailing 啟動於 +1.5R
- `custom_exit`：持倉 8h 無 +0.5R 即平（time stop）
- `leverage`：回傳 20

---

## 關鍵決策（已定）

1. **時間框架**：5m 主決策、15m 確認、1h 過濾噪聲（freqtrade multi-TF）
2. **HITL 模式**：**全階段全自動** — 進場/止損/止盈/熔斷皆 agent 決策執行。TG bot **事後通知 + 管理指令**（`/status /stop /forcesell /profit /balance` 保留 user 介入能力），**不要** 進場前 confirm（避免慢於 MM 節奏）
3. **廣場抓不到**：先跑純 OI+量價版本、Square plugin 降級為 noop
4. **Testnet 週期**：7 天 dry-run + 7 天 binance testnet（共 14 天）
5. **TG Bot**：token `${TG_BOT_TOKEN}`（寫 `.env`，不進 git）
6. **專案焦點**：復現 Lana 交易 agent 邏輯與績效模型（高槓桿短 time-stop + MM 特徵選標 + 固定絕對止損 risk），僅依 100U 本金調整風控尺度，不改策略哲學

---

## 核心設計決定

1. **不從零建框架 → fork freqtrade**。節省 6-8 週。
2. **LLM 不碰選標的，只做 narrative 補充**（plug-in 到 freqtrade 的 feature column）。杜絕「幣安人生」bug。
3. **user Twitter/錢包蒸餾 不進選標鏈路**。只在週報對齊口吻。
4. **P0 禁 LLM**。100U 階段純規則，燒不起 API。
5. **熔斷走 freqtrade 原生 Protections**，不自寫。
6. **TG/告警接 Hermes nia**，不另建。
7. **OI 背離自寫**（無開源庫），但寫成 freqtrade Indicator 標準接口。
8. **Square 爬蟲抽為 Source interface**，抓不到能降級。

---

## 驗收

1. **W0**：notebook p-value + 效應量，user 審閱後才繼續。
2. **W3**：dry-run 7 天報告（PF / Sharpe / max DD）。
3. **W4**：100U 實盤連跑 14 天達升級條件才進 P1。
4. **每 phase git tag** (`p0-live`, `p1-live`, ...) 可回滾。

---

## 交付物

1. `DESIGN.md`（本文定稿 commit）
2. `user_data/strategies/lana_mm_strategy.py` 骨架 + OI 信號
3. `user_data/pairlists/square_hotness.py` 骨架
4. `user_data/notebooks/W0_assumption_validation.ipynb`
5. `configs/config_phase0.json`
6. `docker-compose.yml` (freqtrade + postgres + grafana)
7. `vendored/TradingAgents` submodule
8. 首批 commit push 到 `claude/binance-trading-agent-*`

---

## 關於廣場/X 數據源（user 問）

**Binance 廣場**：**firecrawl 較佳**。理由：
- 廣場頁公開（無需登入），Next.js SSR + JS-heavy → firecrawl 原生處理 JS 頁
- firecrawl 有 **change-tracking** 端點（新帖偵測、不用輪詢整頁）、**search** 端點、**actions**（click 「查看更多」、scroll 載入）
- user 已有 self-host `firecrawl/` + MCP skill（`firecrawl:firecrawl`），零部署成本
- P95 3.4s 延遲 << 60s 信號要求
- 退路：被風控時切 **camofox**（C++ 級指紋偽造更狠）

**X/Twitter**：**camofox-browser**（不是 firecrawl）。理由：
- X 已知嚴格反爬 + 許多內容需登入才可見 → firecrawl 不適用
- **camofox 是為 anti-detection 設計**（Camoufox Firefox fork，指紋在 C++ 層偽造）
- 內建 `@twitter_search` macro 和 session-isolated cookie import → user 可把個人 X cookies 放 `~/.camofox/cookies/x.txt`，走登入態瀏覽
- OpenClaw plugin `@askjo/camofox-browser` 提供 10 個工具（navigate/click/type/snapshot/scroll）
- 實作：`x_source.py` 調 `http://localhost:9377` REST API
  - 抓 CZ / He Yi / 大 V timeline → 被轉發/提及的新幣 ticker
  - `@twitter_search` 抓 ticker cashtag 熱度
  - accessibility snapshot 比 HTML 小 90% → 省 LLM token
- **不需要新 skill**，camofox 可由 `./scripts/setup_camofox.sh` 一鍵 clone+fetch

**用戶 xint MCP** 已標記移除（stdout 污染 + schema 無效，`project_xint_mcp_removed.md`）→ 不採用。

## Sources

- [freqtrade GitHub](https://github.com/freqtrade/freqtrade) — 39.9k★ 交易框架
- [TradingAgents (TauricResearch)](https://github.com/TauricResearch/TradingAgents) — LLM 多 agent
- [Firecrawl](https://github.com/firecrawl/firecrawl) — web scraping, search, change-tracking
- [camofox-browser](https://github.com/jo-inc/camofox-browser) — anti-detection Firefox REST API
- [Binance Open Interest API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- [Binance 永續合約規格](https://www.binance.com/en/support/faq/usd%E2%93%A2-margined-futures-contract-specifications-360033161972)
- [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading)
