"""Aggregate the ratio-spread backtest, rank combos, and headline the tail risk
versus the defined-risk iron fly from backtest.py.

Usage: ratio_report.py [--ticker SPY] [--since 2022-01-01] [--json results/ratio_report.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def metrics(pnl: pd.Series) -> dict:
    cum = pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    ann = np.sqrt(252)
    sharpe = pnl.mean() / pnl.std() * ann if pnl.std() > 0 else 0.0
    return {
        "total": round(float(pnl.sum()), 0),
        "mean_day": round(float(pnl.mean()), 2),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(dd), 0),
        "worst_day": round(float(pnl.min()), 0),
        "best_day": round(float(pnl.max()), 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--since", default="2022-01-01")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ticker = args.ticker.upper()

    df = pd.read_csv(RESULTS / f"{ticker.lower()}_ratio_daily_pnl.csv", index_col=0, parse_dates=True)
    df_full = df.copy()
    if args.since:
        df = df.loc[args.since:]
    label = f"{df.index[0].date()} -> {df.index[-1].date()}"

    ratio_cols = [c for c in df.columns if c.startswith("ratio_")]

    rank = (
        pd.DataFrame({c: metrics(df[c]) for c in ratio_cols})
        .T.sort_values("sharpe", ascending=False)
    )
    print(f"\n=== {ticker} ratio-spread sweep ({label}), per 1 contract set, $ ===")
    print("(mid% = OTM offset for the 2x leg, far% = additional OTM offset for the 1x leg)")
    print(rank.to_string())

    best = rank.index[0]
    combo = df["straddle_pnl"] + df[best]

    print(f"\n=== Components ({label}) ===")
    comp = pd.DataFrame({
        "straddle_only": metrics(df["straddle_pnl"]),
        f"best_ratio ({best})": metrics(df[best]),
        "combined": metrics(combo),
    }).T
    print(comp.to_string())

    # tail risk callout: worst single day/week, full history (not just since-window)
    worst_full = df_full[ratio_cols].min().min()
    worst_full_combo = df_full[ratio_cols].min().idxmin()
    worst_full_date = df_full[ratio_cols].idxmin()[worst_full_combo]
    full_label = f"{df_full.index[0].date()}-present"
    print(f"\n=== TAIL RISK (full history {full_label}, all combos) ===")
    print(f"Worst single day: ${worst_full:,.0f} on {worst_full_date.date()} ({worst_full_combo})")
    print("This structure has NO purchased wing -- loss accelerates to 2x/pt past the mid strike")
    print("and 3x/pt past the far strike. The long LEAPS straddle only partially offsets this")
    print("(different expiry => delta/vega hedge only, no matching daily gamma).")

    print(f"\n=== Combined portfolio P&L aggregations ({best}) ===")
    aggs = {}
    for name, rule in [("weekly", "W-FRI"), ("quarterly", "QE"), ("yearly", "YE")]:
        agg = combo.resample(rule).sum()
        agg = agg[agg != 0]
        aggs[name] = agg
        print(f"\n-- {name} (last 12) --")
        print(agg.tail(12).round(0).to_string())

    if args.json:
        payload = {
            "ticker": ticker,
            "period": label,
            "best_combo": best,
            "rank": rank.reset_index().rename(columns={"index": "combo"}).to_dict("records"),
            "components": comp.reset_index().rename(columns={"index": "leg"}).to_dict("records"),
            "tail_risk": {
                "worst_day": round(float(worst_full), 0),
                "worst_date": str(worst_full_date.date()),
                "worst_combo": worst_full_combo,
            },
            "cum_curves": {
                "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "straddle": df["straddle_pnl"].cumsum().round(0).tolist(),
                "best_ratio": df[best].cumsum().round(0).tolist(),
                "combined": combo.cumsum().round(0).tolist(),
            },
            "yearly": {str(k.year): round(float(v), 0) for k, v in aggs["yearly"].items()},
            "quarterly": {f"{k.year}Q{k.quarter}": round(float(v), 0) for k, v in aggs["quarterly"].items()},
        }
        Path(args.json).write_text(json.dumps(payload))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
