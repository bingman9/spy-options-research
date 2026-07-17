"""Download SPY OHLC, ^VIX, and ^VIX9D daily history and cache to data/*.csv."""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
START = "2016-01-01"


def fetch(symbol: str, name: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, auto_adjust=False, progress=False)
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
    DATA_DIR.mkdir(exist_ok=True)
    fetch("SPY", "spy")
    fetch("^VIX", "vix")
    fetch("^VIX9D", "vix9d")


if __name__ == "__main__":
    sys.exit(main())
