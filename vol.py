"""Shared, ticker-agnostic helpers: realized volatility and strike rounding."""
import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility from a rolling window of daily log returns.

    Used as an implied-vol proxy for any ticker that doesn't have a VIX-style
    index (i.e. everything except SPY/SPX).
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def strike_increment(price):
    """Approximate standard listed-option strike spacing for the given underlying
    price(s). Always returns an ndarray (use float(...) at scalar call sites)."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    return np.select(
        [price < 25, price < 200, price < 500],
        [1.0, 2.5, 5.0],
        default=10.0,
    )


def round_to_increment(value, inc):
    """Round value(s) to the nearest multiple of inc (elementwise)."""
    return np.round(np.asarray(value, dtype=float) / inc) * inc


def round_to_strike(price):
    """Round a price/strike level to the nearest valid increment for its own
    magnitude. Always returns an ndarray (use float(...) at scalar call sites)."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    return round_to_increment(price, strike_increment(price))
