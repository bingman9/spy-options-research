"""Backtest: long 1y ATM SPY straddle (rolled at 90 DTE) + daily short 0DTE iron butterflies.

Outputs results/daily_pnl.csv with one row per trading day:
  straddle mark-to-model P&L, and fly P&L for each (width, offset) combo.
All values are per 1 contract (x100 multiplier) in dollars.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pricing import straddle_price, iron_fly_credit, iron_fly_settlement

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

R = 0.04                 # risk-free rate assumption
MULT = 100               # contract multiplier
ROLL_DTE = 90            # roll LEAPS when 90 calendar days remain
LEAPS_DAYS = 365
FEE_PER_LEG = 0.65       # commission per contract leg
WIDTHS = [5, 10, 15, 20]
OFFSETS = [-5, 0, 5]     # fly center = open strike + offset


def load_data() -> pd.DataFrame:
    spy = pd.read_csv(DATA / "spy.csv", index_col="Date", parse_dates=True)
    vix = pd.read_csv(DATA / "vix.csv", index_col="Date", parse_dates=True)["Close"].rename("vix")
    vix9d = pd.read_csv(DATA / "vix9d.csv", index_col="Date", parse_dates=True)["Close"].rename("vix9d")
    df = spy.join(vix).join(vix9d)
    df["vix"] = df["vix"].ffill()
    df["vix9d"] = df["vix9d"].fillna(df["vix"] * 0.9)
    return df.dropna(subset=["Open", "Close", "vix"])


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
            strike = round(row["Close"] / 5) * 5
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
    row = df.iloc[i]
    dt = df.index[i]
    T = max((expiry - dt).days, 0) / 365
    return float(straddle_price(row["Close"], strike, T, row["vix"] / 100, R))


def fly_legs(df: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L of selling one 0DTE iron fly per day for each (width, offset) combo."""
    T0 = 1 / 252
    out = pd.DataFrame(index=df.index)
    sigma = df["vix9d"].values / 100
    opens = df["Open"].values
    closes = df["Close"].values
    for w in WIDTHS:
        for off in OFFSETS:
            centers = np.round(opens) + off
            credit = iron_fly_credit(opens, centers, w, T0, sigma, R)
            settle = iron_fly_settlement(closes, centers, w)
            pnl = (credit - settle) * MULT - 4 * FEE_PER_LEG
            out[f"fly_w{w}_o{off:+d}"] = pnl
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_data()
    strad = straddle_leg(df)
    flies = fly_legs(df)
    daily = pd.concat([df[["Open", "Close", "vix", "vix9d"]], strad, flies], axis=1)
    daily.to_csv(RESULTS / "daily_pnl.csv")
    rolls = pd.DataFrame(strad.attrs["rolls"], columns=["date", "strike", "expiry"])
    rolls.to_csv(RESULTS / "straddle_rolls.csv", index=False)
    print(f"{len(daily)} trading days {daily.index[0].date()} -> {daily.index[-1].date()}")
    print(f"{len(rolls)} straddle rolls")
    print(f"NaNs: {int(daily.isna().sum().sum())}")
    print(f"wrote {RESULTS / 'daily_pnl.csv'}")


if __name__ == "__main__":
    main()
