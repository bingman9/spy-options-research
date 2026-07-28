"""Backtest: long 1y ATM straddle (unchanged, rolled at 90 DTE) + daily short
0DTE RATIO SPREAD on each side: sell 2 contracts at a "mid" strike + 1 contract
at a farther "third leg" strike, no long wing purchased same day.

Works for ANY ticker (see backtest.py for the VIX-vs-realized-vol and
percent-of-spot generalization notes -- same approach here).

This is NOT a defined-risk structure like the iron fly: past the mid strike the
short book loses 2x underlying per $1 move; past the far strike it loses 3x
(2 mid contracts + 1 far contract all in the money at once). The long 1y
straddle offsets some of this but is a different expiry, so the hedge is
imperfect (delta/vega only, no matching daily gamma).

Sweeps mid_offset (% OTM for the 2x leg) x far_extra (additional % OTM beyond
mid, for the 1x leg), same magnitude applied symmetrically to calls and puts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pricing import ratio_credit, ratio_settlement
from vol import strike_increment, round_to_increment
from backtest import load_data, straddle_leg

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

R = 0.04
MULT = 100
FEE_PER_LEG = 0.65     # 3 legs per side (2x mid + 1x far) -> 6 legs/day total (calls+puts)
MID_OFFSETS = [0.01, 0.02, 0.03]        # mid strike = open * (1 +/- mid_offset)
FAR_EXTRAS = [0.02, 0.04, 0.06]         # far strike = open * (1 +/- (mid_offset + far_extra))


def ratio_legs(df: pd.DataFrame) -> pd.DataFrame:
    T0 = 1 / 252
    out = pd.DataFrame(index=df.index)
    sigma = df["vix9d"].values / 100
    opens = df["Open"].values
    closes = df["Close"].values
    inc = strike_increment(opens)

    for mid_off in MID_OFFSETS:
        for far_extra in FAR_EXTRAS:
            far_off = mid_off + far_extra
            call_mid = round_to_increment(opens * (1 + mid_off), inc)
            call_far = round_to_increment(opens * (1 + far_off), inc)
            put_mid = round_to_increment(opens * (1 - mid_off), inc)
            put_far = round_to_increment(opens * (1 - far_off), inc)

            call_credit = ratio_credit(opens, call_mid, call_far, T0, sigma, R, "call")
            call_settle = ratio_settlement(closes, call_mid, call_far, "call")
            put_credit = ratio_credit(opens, put_mid, put_far, T0, sigma, R, "put")
            put_settle = ratio_settlement(closes, put_mid, put_far, "put")

            total_credit = call_credit + put_credit
            total_settle = call_settle + put_settle
            pnl = (total_credit - total_settle) * MULT - 6 * FEE_PER_LEG

            key = f"ratio_m{int(mid_off*100)}_f{int(far_extra*100)}"
            out[key] = pnl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    args = ap.parse_args()
    ticker = args.ticker.upper()

    df = load_data(ticker)
    strad = straddle_leg(df)
    ratios = ratio_legs(df)
    daily = pd.concat([df[["Open", "Close", "vix", "vix9d"]], strad, ratios], axis=1)
    daily.to_csv(RESULTS / f"{ticker.lower()}_ratio_daily_pnl.csv")

    print(f"{ticker}: {len(daily)} trading days {daily.index[0].date()} -> {daily.index[-1].date()}")
    print(f"NaNs: {int(daily.isna().sum().sum())}")

    ratio_cols = [c for c in ratios.columns]
    worst = daily[ratio_cols].min().min()
    worst_combo = daily[ratio_cols].min().idxmin()
    worst_date = daily[ratio_cols].idxmin()[worst_combo]
    print(f"\nWorst single-day loss across all combos (full history): "
          f"${worst:,.0f} on {worst_date.date()} ({worst_combo})")

    out_path = RESULTS / f"{ticker.lower()}_ratio_daily_pnl.csv"
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
