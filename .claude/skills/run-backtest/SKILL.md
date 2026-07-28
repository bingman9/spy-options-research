---
name: run-backtest
description: Set up and run the options backtests in this repo (pricer sanity checks, iron-fly backtest, ratio-spread backtest, report) for SPY or ANY other ticker. Use when asked to run, verify, or refresh this project's backtests.
---

## Setup

A `.venv` may already exist. If not:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy yfinance
```

## Any ticker, not just SPY

Every script below takes `--ticker SYMBOL` (default `SPY`). Data and results are
namespaced per ticker: `data/<ticker>.csv`, `results/<ticker>_daily_pnl.csv`, etc.
Just fetch the new ticker's data first, then run the same commands with
`--ticker`.

**How it generalizes:**
- **Volatility input:** SPY uses cached real `^VIX`/`^VIX9D` (implied vol) when
  present. Any other ticker falls back to its own 60-day / 20-day realized
  volatility (`vol.py:realized_vol`) as a proxy — there's no VIX-equivalent for
  individual names.
- **Strike width/offset:** expressed as a **percent of that day's spot price**
  (`WIDTH_PCTS` / `OFFSET_PCTS` / `MID_OFFSETS` / `FAR_EXTRAS` in
  `backtest.py` / `ratio_backtest.py`), not a fixed dollar amount — so the same
  parameters are meaningful whether the stock trades at $8 or $3,400.
- **Strike rounding:** adaptive increment by price level via
  `vol.strike_increment` (≈ real listed-option spacing: $1 under $25, $2.50
  under $200, $5 under $500, $10 above).

## Refresh data

```bash
.venv/bin/python fetch_data.py --ticker SPY
# → SPY: N rows 2016-01-04 -> <latest> -> data/spy.csv (+ vix.csv, vix9d.csv for SPY only)

.venv/bin/python fetch_data.py --ticker AAPL
# → AAPL: N rows 2016-01-04 -> <latest> -> data/aapl.csv
```

## Sanity-check the pricer (ticker-independent)

```bash
.venv/bin/python pricing.py
# → ATM 1y call @20% vol: 9.9251 (9.93% of spot)  put: 6.0040
# → put-call parity gap: 0.00e+00
# → ratio settlement slopes: 2x between mid/far, 3x beyond far strike -- confirmed
# → all pricer sanity checks pass
```

## Run the backtests

Defined-risk iron-butterfly version (long 1y ATM straddle + daily short iron fly,
width/offset sweep):

```bash
.venv/bin/python backtest.py --ticker SPY
# → SPY: N trading days 2016-01-04 -> <latest>
# → 15 straddle rolls
# → NaNs: 0
# → wrote results/spy_daily_pnl.csv
```

Uncapped-risk ratio-spread version (same long straddle + daily short 2x-mid/1x-far
ratio spread, no purchased wing):

```bash
.venv/bin/python ratio_backtest.py --ticker SPY
# → SPY: N trading days 2016-01-04 -> <latest>
# → NaNs: 0
# → Worst single-day loss across all combos (full history): $-12,288 on 2025-04-09 (ratio_m1_f2)
# → wrote results/spy_ratio_daily_pnl.csv
```

Same commands with `--ticker AAPL` (or any other symbol) produce
`results/aapl_daily_pnl.csv` etc. without touching the SPY results.

## Generate the report

```bash
.venv/bin/python ratio_report.py --ticker SPY --since 2022-01-01 --json results/ratio_report.json
```

Prints the parameter sweep (ranked by Sharpe), component breakdown
(straddle-only / best-combo / combined), a tail-risk callout (worst single day,
full history), and weekly/quarterly/yearly P&L tables. Writes the same data as
JSON to the `--json` path for feeding into a chart/report template.

## Notes

- All scripts are self-contained CLI runs — no server, no long-lived process.
- `fetch_data.py` must run before the backtests the first time for a given
  ticker (or whenever you want fresher data); the backtests just read the
  cached CSVs in `data/`.
- `ratio_report.py --since` accepts any date; omit it to use full history from
  whenever the data starts. Real daily-expiry SPY options only exist since
  ~2022, so `--since 2022-01-01` is the realistic window for SPY; for other
  tickers, weekly/daily options availability varies — treat `--since` as a
  research choice, not a hard constraint the model enforces.
- This is a Black-Scholes model, not real option quotes, for any ticker — flat
  vol ignores the smile, and assumes an idealized options market with whatever
  expiries the model wants (many stocks don't actually have 0DTE or 1-year
  LEAPS liquidity). Research only, not trading advice.
