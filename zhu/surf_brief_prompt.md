# Zhu Surf Brief — Hermes cron prompt

## One-line install + schedule

```bash
# 1. install surf skill into Hermes
hermes skills install asksurf-ai/surf-skills --yes

# 2. schedule daily 08:00 brief (cron format: min hour dom mon dow)
hermes cron create "0 8 * * *" \
  --name "zhu-surf-brief" \
  --skill "surf-skills" \
  --deliver telegram \
  "$(cat /Users/51mini/binance-trading-agent/zhu/surf_brief_prompt.md)"
```

## Prompt body (Hermes reads this as task each 08:00)

You are producing the Zhu morning alpha brief for Solana memes. Run the following searches using the surf skill:

1. `CoinMarketCap Solana gainers 24h top 10`
2. `Solana meme trending twitter last 24h`
3. `pump.fun graduating tokens past 24h`
4. `Bitget new listing Solana 24h`
5. `social driven crypto community surge 2026`

Synthesize findings into a JSON file at exactly this path:
`/Users/51mini/binance-trading-agent/zhu/surf_brief.json`

Schema (strict):
```json
{
  "generated_at": <unix_epoch_int>,
  "themes": ["theme_1", "theme_2"],
  "candidates": [
    {
      "ca": "<Solana contract address, base58 32-44 chars>",
      "symbol": "<TICKER>",
      "reason": "<why this CA matters, 1 sentence>",
      "source": "<where found: cmc/twitter/pumpfun/bitget/other>"
    }
  ]
}
```

Rules (non-negotiable):
- Only include tokens with verifiable Solana contract addresses. No CA = skip.
- Dedupe by CA.
- Max 15 candidates.
- Prefer low-mc (< $5M) with recent social spike or smart-money inflow.
- No stablecoins, no top-100 established tokens.

After writing the file, send to Telegram a 3-5 line summary of the themes + top 3 candidates. Include the exact CAs so user can cross-check.
