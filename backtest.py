"""Backtest: long 1y ATM straddle (rolled at 90 DTE) + daily short 0DTE iron butterflies.

Works for ANY ticker. For SPY specifically, uses cached ^VIX/^VIX9D (real implied-vol
indices) when available; for any other ticker, falls back to that ticker's own
60-day / 20-day realized volatility as a proxy.

Wing width and center offset are expressed as a PERCENT of the day's spot price
(not a fixed dollar amount), so the same parameters are sensible whether the
ticker trades at $10 or $2,000. Strikes are rounded to an adaptive increment
based on price level (see vol.strike_increment).

Outputs results/<ticker>_daily_pnl.csv with one row per trading day: straddle
mark-to-model P&L, and fly P&L for each (width%, offset%) combo. All values are
per 1 contract (x100 multiplier) in dollars.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pricing import straddle_price, iron_fly_credit, iron_fly_settlement
from vol import realized_vol, strike_increment, round_to_increment, round_to_strike

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

R = 0.04                 # risk-free rate assumption
MULT = 100               # contract multiplier
ROLL_DTE = 90            # roll LEAPS when 90 calendar days remain
LEAPS_DAYS = 365
FEE_PER_LEG = 0.65       # commission per contract leg
WIDTH_PCTS = [0.01, 0.02, 0.03, 0.04]     # fly wing width, as % of spot
OFFSET_PCTS = [-0.007, 0.0, 0.007]        # fly center offset from spot, as %


def load_data(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{ticker.lower()}.csv", index_col="Date", parse_dates=True)
    df = df.dropna(subset=["Open", "Close"])

    vix_path, vix9d_path = DATA / "vix.csv", DATA / "vix9d.csv"
    if ticker.upper() == "SPY" and vix_path.exists() and vix9d_path.exists():
        vix = pd.read_csv(vix_path, index_col="Date", parse_dates=True)["Close"].rename("vix")
        vix9d = pd.read_csv(vix9d_path, index_col="Date", parse_dates=True)["Close"].rename("vix9d")
        df = df.join(vix).join(vix9d)
        df["vix"] = df["vix"].ffill()
        df["vix9d"] = df["vix9d"].fillna(df["vix"] * 0.9)
    else:
        df["vix"] = realized_vol(df["Close"], 60) * 100
        df["vix9d"] = realized_vol(df["Close"], 20) * 100
        df["vix"] = df["vix"].bfill()
        df["vix9d"] = df["vix9d"].bfill()

    return df.dropna(subset=["Open", "Close", "vix", "vix9d"])


def straddle_leg(df: pd.DataFrame) -> pd.DataFrame:
    """Mark-to-model daily P&L of the rolled 1y ATM straddle (1 contract)."""
    dates = df.index
    marks = np.zeros(len(dates))
    entries = []  # (date, strike, expiry)
    strike = expiry = None
    prev_mark = None
    pnl = np.zeros(len(dates))

    for i, (dt, row) in enumerate(df.iterrows()):
        if expiry is None or (expiry - dt).days <= ROLL_DTE:
            strike = round_to_strike(row["Close"]).item()
            expiry = dt + pd.Timedelta(days=LEAPS_DAYS)
            entries.append((dt.date(), strike, expiry.date()))
            new_mark = float(
                straddle_price(row["Close"], strike, LEAPS_DAYS / 365, row["vix"] / 100, R)
            )
            if prev_mark is not None:
                pnl[i] = (marks_prev_close(df, i, strike, expiry) - prev_mark) * MULT
            pnl[i] -= 4 * FEE_PER_LEG
            prev_mark = new_mark
            marks[i] = new_mark
            continue
        T = max((expiry - dt).days, 0) / 365
        mark = float(straddle_price(row["Close"], strike, T, row["vix"] / 100, R))
        marks[i] = mark
        pnl[i] = (mark - prev_mark) * MULT
        prev_mark = mark

    out = pd.DataFrame({"straddle_mark": marks, "straddle_pnl": pnl}, index=dates)
    out.attrs["rolls"] = entries
    return out


def marks_prev_close(df: pd.DataFrame, i: int, strike: float, expiry: pd.Timestamp) -> float:
    """Value of the outgoing straddle at today's close (just before rolling)."""
    row = df.iloc[i]
    dt = df.index[i]
    T = max((expiry - dt).days, 0) / 365
    return float(straddle_price(row["Close"], strike, T, row["vix"] / 100, R))


def fly_legs(df: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L of selling one 0DTE iron fly per day for each (width%, offset%) combo."""
    T0 = 1 / 252
    out = pd.DataFrame(index=df.index)
    sigma = df["vix9d"].values / 100
    opens = df["Open"].values
    closes = df["Close"].values
    inc = strike_increment(opens)

    for w_pct in WIDTH_PCTS:
        for off_pct in OFFSET_PCTS:
            centers = round_to_increment(opens * (1 + off_pct), inc)
            widths = np.maximum(round_to_increment(opens * w_pct, inc), inc)
            credit = iron_fly_credit(opens, centers, widths, T0, sigma, R)
            settle = iron_fly_settlement(closes, centers, widths)
            pnl = (credit - settle) * MULT - 4 * FEE_PER_LEG
            key = f"fly_w{w_pct*100:g}pct_o{off_pct*100:+.1f}pct"
            out[key] = pnl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    args = ap.parse_args()
    ticker = args.ticker.upper()

    RESULTS.mkdir(exist_ok=True)
    df = load_data(ticker)
    strad = straddle_leg(df)
    flies = fly_legs(df)
    daily = pd.concat([df[["Open", "Close", "vix", "vix9d"]], strad, flies], axis=1)
    daily.to_csv(RESULTS / f"{ticker.lower()}_daily_pnl.csv")
    rolls = pd.DataFrame(strad.attrs["rolls"], columns=["date", "strike", "expiry"])
    rolls.to_csv(RESULTS / f"{ticker.lower()}_straddle_rolls.csv", index=False)
    print(f"{ticker}: {len(daily)} trading days {daily.index[0].date()} -> {daily.index[-1].date()}")
    print(f"{len(rolls)} straddle rolls")
    print(f"NaNs: {int(daily.isna().sum().sum())}")
    print(f"wrote {RESULTS / f'{ticker.lower()}_daily_pnl.csv'}")


if __name__ == "__main__":
    main()
