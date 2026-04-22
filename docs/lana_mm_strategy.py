"""LanaMMStrategy — 復刻 Lana (@lanaaielsa, 棍哥) 的鏈上長尾打法.

Lana 本人引述 (OKX 採訪):
  "十次裡九次是錯的, 交易是忍受低勝率的枯燥, 和對人性的管理."
  "100u 起, 10 天跑到 20w u, 2000 倍."
  核心能力 = 鏈上數據信號 + 管理人性情緒

W0 統計驗證 (d=0.1 + MWU 不顯著) 正好吻合這 pattern:
  - 大多數交易平庸或虧
  - 少數長尾大勝拉高均值
  - 勝率 <10%, 單筆期望正但方差極大

本策略策略論題:
  H1. MM 建倉留 OI 足跡 (已驗證 p<0.05, d 偏弱但 edge 來自長尾)
  H2. 廣場情緒爆量 → 領先訊號 (待 7 天歷史累積)
  H3. H1 ∩ H2 疊加 (未來)

執行哲學 (照抄 Lana 核心, 加 100U 本金護欄):
  • 高槓桿 + 短時間框 (P0 20x / 6h time-stop)
  • 固定 % 風險 1% (她是固定 $, 100U 照抄爆倉)
  • 敢於追高: 止損近 (-3%), time-stop 快 (6h)
  • 同時只一個倉 (P0), 避免相關性爆倉
  • 連敗 3 次 24h 冷卻 (防情緒化爆損)
  • 熔斷兜底 (日內 -5% 停 48h)

Harness 護欄 (不破壞 edge, 只擋陷阱):
  • funding 極端過濾: |funding| < 0.01% (8h) 才進 — 避免 pump-dump 尾追高
  • BTC 4h 趨勢過濾: BTC 4h close > MA(24) 才做 alt long — macro 逆風不硬抗
  • entry_score + z_oi + z_px + vol_struct 全 AND 條件追 precision

LLM 在 P0 禁用 (qwen 判斷能力不足). P1+ 透過 `narrative_score` column 注入.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# freqtrade 直接以檔案路徑載入策略, 需手動把 user_data 的父目錄塞進 sys.path,
# user_data 才能當 top-level package 讓 signals 子模組可被 import
_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from freqtrade.strategy import (  # noqa: E402
    IStrategy,
    IntParameter,
    DecimalParameter,
)

# LLM for dynamic exit re-evaluation (Lana 15min Cron 風格)
from user_data.signals.external_data import apply_externals  # noqa: E402
from user_data.signals.funding_anomaly import funding_anomaly, short_squeeze_setup  # noqa: E402
from user_data.signals.oi_divergence import oi_build_up_flag, oi_price_divergence  # noqa: E402
from user_data.signals.sentiment import apply_sentiment  # noqa: E402
from user_data.signals.momentum_trigger import momentum_path_b  # noqa: E402
from user_data.signals.taker_imbalance import taker_imbalance  # noqa: E402
from user_data.signals.volume_structure import volume_compression  # noqa: E402


class LanaMMStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    informative_timeframes = ["15m", "1h"]

    can_short = True  # P0 只做 long; Lana 初期全 long。留後門。
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 300

    # --- hyperoptable 權重 (重構: OI 提成 Layer-1 硬條件, 不再進融合分) ---
    # confirmation_score = Layer-3 輔助信號, 給 OI+harness 通過後的強度加分
    w_oi = DecimalParameter(0.0, 0.2, default=0.0, space="buy", optimize=False)  # 禁用, OI 已在 Layer 1
    w_funding = DecimalParameter(0.0, 0.6, default=0.2, space="buy", optimize=True)
    w_taker = DecimalParameter(0.0, 0.6, default=0.3, space="buy", optimize=True)
    w_volume = DecimalParameter(0.0, 0.6, default=0.3, space="buy", optimize=True)
    w_sentiment = DecimalParameter(0.0, 1.0, default=0.4, space="buy", optimize=True)

    # --- 3 層門檻 (調校目標: 2-7 trade/week) ---
    # Layer 1: Lana 核心 OI 背離 (必要條件 AND)
    # 放寬到早前測試過 TAO 實際觸發的 z_oi~2.2 水準
    oi_div_min = DecimalParameter(1.5, 3.5, default=2.0, space="buy", optimize=True)
    z_oi_min = DecimalParameter(2.0, 4.0, default=2.2, space="buy", optimize=True)
    z_px_max = DecimalParameter(0.3, 1.5, default=1.0, space="buy", optimize=True)

    # Layer 2: Harness (必要條件 AND, 過濾陷阱環境)
    # (funding_abs_max / btc_trend_up 在 populate_entry_trend 實作)

    # Layer 3: Confirmation score (軟門檻, 加分通過才進, 不含 OI 避免重複)
    # 預設 0 = 只要正分就 OK (不要嚴到堵)
    confirmation_min = DecimalParameter(-0.5, 1.5, default=0.0, space="buy", optimize=True)
    vol_struct_min = DecimalParameter(-1.0, 2.0, default=-0.5, space="buy", optimize=True)
    # 放寬 funding filter: 0.01% 太嚴, alt 常 0.03-0.05%, 只擋極端 >0.1%
    funding_abs_max = DecimalParameter(0.0001, 0.002, default=0.0005, space="buy", optimize=True)
    time_stop_hours = IntParameter(6, 48, default=24, space="sell", optimize=True)  # 24h 老幣 (棍哥 meme 1-3d 抱)

    # --- Lana 風格硬 stop (custom_stoploss 覆蓋) ---
    stoploss = -0.03  # doc: hard stop 3% (docs/STRATEGY.md Exit §1)
    trailing_stop = False  # 自寫 trailing 邏輯

    # --- time-stop guard ---
    minimal_roi = {"0": 100}  # 禁 ROI 賣出；全由 custom_exit + trailing 管

    # --- Position pyramid (棍哥 winner 加碼) ---
    position_adjustment_enable = True
    max_entry_position_adjustment = 2  # base + 2 加倉 = max 3 層

    # --- 槓桿 per-pair dynamic (讀 leverage_tiers_USDT.json) ---
    # Lana 哲學: meme 高槓桿吃尾部, majors 低槓桿避狙擊
    leverage_value = 20  # fallback if tier cache miss
    MEME_PAIRS = {"ORDI/USDT:USDT", "1000SATS/USDT:USDT", "PEPE/USDT:USDT",
                  "DOGE/USDT:USDT", "PNUT/USDT:USDT", "SKYAI/USDT:USDT",
                  "SHIB/USDT:USDT", "WIF/USDT:USDT", "BONK/USDT:USDT"}
    MAJOR_PAIRS = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    _tier_cache: dict = {}  # pair → tier-1 maxLeverage (loaded lazily)

    # --- 新幣判定 (< 30 天, 寬止損/寬 time-stop) ---
    # Lana 訪談: 新幣高開低走, 舊止損全被打. 給寬容度換穩定趨勢
    NEW_LISTING_DAYS = 30
    _pair_age_cache: dict = {}  # pair → is_new (bool)

    # --- Protections (docs/STRATEGY.md §Protections 規格) ---
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 288,   # 24h
                "trade_limit": 6,                  # 放寬: 棍哥 <10% 勝率需更多嘗試
                "stop_duration_candles": 72,       # 放寬: 6h 冷卻而非 24h 熔斷
                "only_per_pair": False,            # doc: global
            },
            {"method": "CooldownPeriod", "stop_duration_candles": 12},  # doc: 1h
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 576,    # 48h
                "trade_limit": 5,
                "stop_duration_candles": 576,      # doc: pause 48h
                "max_allowed_drawdown": 0.05,      # doc: 5% drawdown
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 5760,   # 20 days
                "trade_limit": 2,
                "stop_duration_candles": 10080,    # 35 days
                "required_profit": 0.0,            # doc: PF<1.0 ≡ profit<=0
            },
        ]

    # --- 新幣偵測 (Lana: 新幣高開低走, 用寬止損) ---
    def _macro_risk_on(self) -> bool:
        """Read Redis `macro_regime` key, return True if risk_on (default True if Redis miss)."""
        try:
            import json as _json
            import os as _os
            import redis as _redis
            r = _redis.Redis(
                host=_os.getenv("REDIS_HOST", "redis"),
                port=int(_os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_timeout=1.0,
            )
            raw = r.get("macro_regime")
            if raw:
                return bool(_json.loads(raw).get("risk_on", True))
        except Exception:
            pass
        return True

    def _is_new_listing(self, pair: str) -> bool:
        """CCXT market 資料的 listing time 判斷 < 30 天."""
        if pair in self._pair_age_cache:
            return self._pair_age_cache[pair]
        is_new = False
        try:
            exchange = self.dp._exchange._api  # CCXT instance
            market = exchange.market(pair)
            info = market.get("info", {})
            # Bitget returns launchTime in ms; fallback keys for safety
            launch_ms = int(info.get("launchTime", 0) or info.get("onboardDate", 0) or 0)
            if launch_ms:
                age_days = (time.time() * 1000 - launch_ms) / 86400000
                is_new = age_days < self.NEW_LISTING_DAYS
        except Exception as e:
            logging.getLogger(__name__).debug("_is_new_listing %s err: %s", pair, e)
        self._pair_age_cache[pair] = is_new
        return is_new

    # ==========================================================
    # Indicators
    # ==========================================================
    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- 從 Binance public REST 合併 OI / funding / taker ---
        # (Bitget 只給 OHLCV; Lana edge 依賴 Binance 深流動性的 OI 足跡)
        df = apply_externals(df, metadata["pair"])

        # --- BTC 4h 趨勢 (macro regime filter) ---
        # 簡化版: 取當下 pair 自己的 4h close > MA(24) 作為代替 (避免新增 informative pair 複雜度)
        # 注: 理想是 BTC 4h 趨勢, 但目前 dataframe 沒 BTC 數據.
        # 折衷: 用 5m × 48 bar (= 4h 的 EMA) 作為本 pair 中期趨勢代理,
        # 大部分 alt 與 BTC 高度相關, 也能濾掉 pair 自身的深空頭行情
        ema_4h = df["close"].ewm(span=48, min_periods=24).mean()
        df["btc_trend_up"] = (df["close"] > ema_4h).astype(int)

        # --- 核心 Lana 信號 ---
        from user_data.signals.zscore import rolling_zscore
        _oi_ret = df["open_interest"].pct_change(fill_method=None).fillna(0) if "open_interest" in df.columns else pd.Series(0.0, index=df.index)
        _px_ret = df["close"].pct_change(fill_method=None).fillna(0)
        _win = 48 * 12
        df["z_oi"] = rolling_zscore(_oi_ret, _win).fillna(0.0)
        df["z_px"] = rolling_zscore(_px_ret.abs(), _win).fillna(0.0)
        df["oi_div"] = df["z_oi"] - df["z_px"]
        df["oi_flag"] = oi_build_up_flag(df, "oi_div", threshold=1.5)

        # --- WolfyXBT 風 OI 比值快訊 ---
        # 1h 視窗 (5m × 12 bar) 的 OI%/Price% 比值
        # 例: OI +20% / Price +3% = 6.7 → 主動 MM 還沒推動價格
        if "open_interest" in df.columns:
            _oi_1h = df["open_interest"].pct_change(12).fillna(0) * 100
            _px_1h = df["close"].pct_change(12).fillna(0) * 100
            _safe_px = _px_1h.where(_px_1h.abs() > 0.5, 0.5)  # 避除以 0, 價格幾乎沒動才看 OI
            df["oi_fast_ratio"] = (_oi_1h / _safe_px).clip(-20, 20).fillna(0.0)
            df["oi_fast_flag"] = ((df["oi_fast_ratio"] > 5.0) & (_px_1h.abs() < 5.0)).astype(int)
        else:
            df["oi_fast_ratio"] = 0.0
            df["oi_fast_flag"] = 0

        df["funding_anom"] = funding_anomaly(df)
        df["squeeze_flag"] = short_squeeze_setup(df)

        df["taker_z"] = taker_imbalance(df)
        df["vol_struct"] = volume_compression(df)

        # --- 情緒 + Bitget 原生 OI 從 Redis 注入 ---
        df = apply_sentiment(df, metadata["pair"])
        # Fallback: 若 Binance OI 缺（z_oi=0）且 Bitget OI 有資料 → 用 Bitget
        if "bitget_oi_z" in df.columns:
            _bz = df["bitget_oi_z"].fillna(0.0)
            df["z_oi"] = df["z_oi"].where(df["z_oi"] != 0.0, _bz)

        # --- Confirmation score (Layer 3, 不含 OI 避免與 Layer 1 重複) ---
        # 這是「OI 背離通過後, 其他信號是否配合」的加分項
        df["confirmation_score"] = (
            self.w_funding.value * df["funding_anom"]
            + self.w_taker.value * df["taker_z"]
            + self.w_volume.value * df["vol_struct"]
            + self.w_sentiment.value * df["sentiment_score"]
        )
        # 保留 entry_score 別名供舊程式碼相容
        df["entry_score"] = df["confirmation_score"]

        # --- time-stop reference ---
        df["entry_price_ref"] = df["close"]

        # ====== P1 Shadow Path B (log only, no entry impact) ======
        try:
            from user_data.signals.sentiment import sentiment_features
            _feats = sentiment_features(metadata["pair"])
            df["path_b_signal"] = momentum_path_b(df, _feats).astype(int)
            # Rising edge → log to Redis + Kronos + LLM debate
            if len(df) >= 2 and df["path_b_signal"].iloc[-1] == 1 and df["path_b_signal"].iloc[-2] == 0:
                self._log_shadow_signal(metadata["pair"], df.iloc[-1], _feats)
                # Kronos shadow: 預測下 75min (15 bar)
                try:
                    from user_data.utils.kronos_predictor import predict as _kp, log_shadow as _klog
                    _kres = _kp(metadata["pair"], df.tail(200), pred_len=15)
                    _klog(metadata["pair"], _kres, float(df["close"].iloc[-1]))
                    df.loc[df.index[-1], "kronos_up_prob"] = _kres.up_prob
                    df.loc[df.index[-1], "kronos_expected_return"] = _kres.expected_return
                except Exception:
                    pass
                # LLM Bull-Bear debate (TradingAgents 風，1 回合，Redis 15min cache)
                try:
                    from user_data.signals.llm_debate import debate as _deb
                    _dres = _deb(metadata["pair"], {
                        "hotness_score": float(_feats.get("square_ticker_mention_z", 0) or 0),
                        "kol_boost": float(_feats.get("x_kol_boost", 0) or 0),
                        "cg_trending": float(_feats.get("coingecko_trending_score", 0) or 0),
                        "bitget_gainer_rank": float(_feats.get("bitget_gainer_rank", 0) or 0),
                        "funding_rate": float(df["funding_rate"].iloc[-1] if "funding_rate" in df else 0),
                        "btc_trend_up": int(df["btc_trend_up"].iloc[-1] if "btc_trend_up" in df else 1),
                        "z_oi": float(df["z_oi"].iloc[-1] if "z_oi" in df else 0),
                        "macro_risk_on": self._macro_risk_on(),
                    })
                    df.loc[df.index[-1], "llm_conviction"] = _dres.conviction
                    df.loc[df.index[-1], "llm_verdict"] = _dres.verdict
                except Exception:
                    pass
        except Exception as _e:
            df["path_b_signal"] = 0

        return df

    # ==========================================================
    # Entry
    # ==========================================================
    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 三層邏輯 (Lana 哲學為主):
        #
        # Layer 1 (Lana 核心 OI 背離 + 追高) — 不滿足絕不進場
        #   z_oi > 3.0 AND z_px < 0.8 AND oi_div > 2.5
        #   close > MA(12)  (Lana 追高)
        #
        # Layer 2 (Harness 避陷阱) — 全部必須通過
        #   |funding| < 0.05%   (不追 pump 尾)
        #   btc_trend_up        (macro 順風)
        #   volume > 0
        #
        # Layer 3 (Confirmation score 加分) — 其他信號配合才進
        #   confirmation_score > 0.3 (軟, 調整用)
        #   vol_struct > 0 (基本量價配合)
        #
        # 防禦: 空 candle / column miss 時預設 0
        for col in ["z_oi", "z_px", "oi_div", "entry_score", "confirmation_score",
                    "vol_struct", "funding_rate", "btc_trend_up", "path_b_signal"]:
            if col not in df.columns:
                df[col] = 0.0
        if len(df) == 0:
            return df

        # Primary: Path B hotness trigger (Lana 核心哲學)
        #   path_b_signal 由 populate_indicators 經 momentum_path_b() 計算
        #   = (24h pct > 5%) AND (vol_z > 1) AND (close > VWAP) AND (mention_z > 1)
        path_b_mask = df["path_b_signal"] == 1

        # Harness: 風險閘 (保留 — 極端 funding / BTC 下跌 / 空量不進)
        # Macro regime gate — 放寬：僅極端 risk-off (VIX>35 / Fed 超急) 才擋
        # BTC trend gate 移除 (棍哥高頻打法, 熊市也進 meme 反彈)
        # funding filter 放寬 0.0005 → 0.001
        macro_ok = self._macro_risk_on()
        harness = (
            (df["funding_rate"].abs() < max(self.funding_abs_max.value, 0.001))
            & (df["volume"] > 0)
            & macro_ok
        )

        # Optional OI confirm (不阻擋，只加 tag 供 shadow PnL 分群分析)
        oi_confirm = (
            (df["z_oi"] > self.z_oi_min.value)
            & (df["z_px"] < self.z_px_max.value)
            & (df["oi_div"] > self.oi_div_min.value)
            & (df["close"] > df["close"].rolling(12).mean())
        ).fillna(False)

        entry_mask = path_b_mask & harness

        # LLM Bull-Bear debate gate (僅擋明確 skip，conv < 0.25 才剃；missing/error pass-through)
        if "llm_conviction" in df.columns:
            debate_ok = df["llm_conviction"].fillna(1.0) >= 0.25
            entry_mask = entry_mask & debate_ok

        df.loc[entry_mask, "enter_long"] = 1
        df.loc[entry_mask & oi_confirm, "enter_tag"] = "hotness+oi+debate"
        df.loc[entry_mask & ~oi_confirm, "enter_tag"] = "hotness+debate"
        return df

    # ==========================================================
    # Exit
    # ==========================================================
    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 主要出場走 custom_stoploss + custom_exit，這裡只做 signal 反轉保險絲
        df.loc[
            (df["oi_div"] < -1.0) & (df["entry_score"] < 0),
            "exit_long",
        ] = 1
        return df

    # confirm_trade_entry 移除 — qwen 判讀能力不足, 風控交規則層 + freqtrade Protections

    # ==========================================================
    # Custom stoploss — hard + trailing
    # ==========================================================
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        # 新幣分支保留 Lana 原訪談: 新幣高開低走, 舊止損全被打
        is_new = self._is_new_listing(pair)
        if is_new:
            hard = -0.08           # 新幣 8% (Lana 原口述)
            trailing_start = 0.04
            trailing_lock = 0.6     # 60% 給暴漲震盪空間
        else:
            # 老幣棍哥風格: stop 3% 硬, trailing +2% / 60% lock (放寬抓尾部)
            hard = -0.03
            trailing_start = 0.02
            trailing_lock = 0.6

        if current_profit > trailing_start:
            trailing = -(current_profit * trailing_lock)
            return max(hard, trailing)
        return hard

    # ==========================================================
    # Custom exit — time-stop + ladder TP
    # ==========================================================
    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """docs/STRATEGY.md §Exit 優先序:
        1. 交易所硬止損 (custom_stoploss 管)
        2. TP ladder: +9% tp_5R / +6% held>=30min tp_3R_trail
        3. Trailing (custom_stoploss 管, +1.5% 啟動 / 40% lock)
        4. Time stop: 6h 老幣 / 24h 新幣, profit < 0.5%
        5. Signal reversal (populate_exit_trend 管)
        """
        held = (current_time - trade.open_date_utc).total_seconds() / 3600
        is_new = self._is_new_listing(pair)

        # time-stop 新幣 24h / 老幣 6h
        time_stop_hrs = 24 if is_new else self.time_stop_hours.value

        # TP ladder — 棍哥 meme 1-3 日抱暴漲哲學 (讓 winner 飛更久):
        #   +50% moonshot hard cap 防極端回吐
        #   +25% held >= 1h: 交還 trailing (+2% start / 60% lock) 接管 winner
        if current_profit >= 0.50:
            return "tp_moonshot"
        if current_profit >= 0.25 and held >= 1.0:
            return "tp_20R_trail"

        # Time stop (doc Exit §4)
        if held >= time_stop_hrs and current_profit < 0.005:
            return "time_stop_no_progress"

        return None

    def _log_shadow_signal(self, pair: str, row, feats: dict) -> None:
        """寫 Redis: shadow_path_b:{pair}:{ts} 紀錄假設進場時的 close 與訊號 components."""
        try:
            import json
            import os
            import redis as _redis
            r = _redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            ts = int(pd.Timestamp(row["date"]).timestamp())
            payload = {
                "pair": pair,
                "entry_close": float(row["close"]),
                "entry_ts": ts,
                "pct_change_24h": float(row["close"]) and float((row.get("close", 0))),
                "components": {
                    "mention_z": float(feats.get("square_ticker_mention_z", 0) or 0),
                    "cg_score": float(feats.get("coingecko_trending_score", 0) or 0),
                    "sentiment_score": float(feats.get("sentiment_score", 0) or 0) if "sentiment_score" in feats else 0,
                },
            }
            r.setex(f"shadow_path_b:{pair}:{ts}", 86400, json.dumps(payload))
            import logging
            logging.getLogger(__name__).info(
                f"[shadow-pathb] {pair} fired at {row['close']:.4f}, mention_z={payload['components']['mention_z']:.2f}"
            )
        except Exception as _e:
            pass  # Redis 沒接也不影響策略

    # ==========================================================
    # Dynamic stake — 1% equity risk / stop%，貼 min notional
    # ==========================================================
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        equity = self.wallets.get_total("USDT") if self.wallets else proposed_stake
        risk_budget = equity * 0.01  # 1% equity
        stop_pct = 0.03
        notional = risk_budget / stop_pct  # 名義倉位
        stake = notional / leverage  # 保證金
        stake = max(stake, min_stake or 0)
        stake = min(stake, max_stake or stake)
        # 貼 binance alt min notional 5 USDT
        if notional < 5:
            return min_stake or 5 / leverage
        return float(np.round(stake, 2))

    # ==========================================================
    # Pyramid 加倉 (棍哥 winner 加碼哲學)
    # ==========================================================
    def adjust_trade_position(
        self,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """棍哥 pyramid-up:
          - Base 1x + 價格漲 10% 加 0.5x + 漲 20% 再加 0.25x
          - Max 2 次加倉（base + 2 = 3 層）
          - Stop 仍以首層 entry 算（freqtrade 預設）
        """
        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            n_layers = len(filled_entries)
        except Exception:
            return None
        if n_layers >= 1 + self.max_entry_position_adjustment:
            return None
        base_stake = float(filled_entries[0].cost) if filled_entries else 0.0
        if base_stake <= 0:
            return None
        if n_layers == 1 and current_profit >= 0.10:
            return base_stake * 0.5
        if n_layers == 2 and current_profit >= 0.20:
            return base_stake * 0.25
        return None

    # ==========================================================
    # Leverage
    # ==========================================================
    def _load_tier_cache(self) -> None:
        """Lazy-load Bitget leverage tiers (tier-1 maxLeverage per pair)."""
        if self._tier_cache:
            return
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "bitget", "futures", "leverage_tiers_USDT.json")
        try:
            with open(path) as f:
                raw = json.load(f)
            for symbol, tiers in raw.get("data", {}).items():
                if tiers and isinstance(tiers, list):
                    self._tier_cache[symbol] = float(tiers[0].get("maxLeverage", 20))
        except Exception:
            pass

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        self._load_tier_cache()
        tier_max = self._tier_cache.get(pair, max_leverage)
        # Lana-style per-pair target (with 50% tier safety margin)
        if pair in self.MEME_PAIRS:
            target = min(25.0, tier_max * 0.5)
        elif pair in self.MAJOR_PAIRS:
            target = min(15.0, tier_max * 0.3)
        else:
            target = min(20.0, tier_max * 0.4)
        return min(target, max_leverage)
