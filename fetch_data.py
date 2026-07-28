"""Download daily OHLC for any ticker and cache to data/<ticker>.csv.

For SPY specifically, also fetches ^VIX/^VIX9D (real implied-vol indices) so the
backtests can use them instead of the realized-vol fallback used for other tickers.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
START = "2016-01-01"


def fetch(symbol: str, name: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]]
    df.index.name = "Date"
    out = DATA_DIR / f"{name}.csv"
    df.to_csv(out)
    print(f"{symbol}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()} -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY", help="Underlying ticker, e.g. SPY, AAPL, TSLA")
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    ticker = args.ticker.upper()
    fetch(ticker, ticker.lower(), args.start)
    if ticker == "SPY":
        fetch("^VIX", "vix", args.start)
        fetch("^VIX9D", "vix9d", args.start)


if __name__ == "__main__":
    sys.exit(main())
