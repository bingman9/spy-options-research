"""Black-Scholes pricing for European calls/puts (vectorized over numpy arrays)."""
import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, sigma, r=0.04, kind="call"):
    """Price. T in years, sigma annualized. Handles T<=0 as intrinsic value."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    intrinsic = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    T_safe = np.where(T > 0, T, 1e-12)
    sig_safe = np.where(sigma > 0, sigma, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sig_safe**2) * T_safe) / (sig_safe * np.sqrt(T_safe))
    d2 = d1 - sig_safe * np.sqrt(T_safe)
    if kind == "call":
        val = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    else:
        val = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(T > 0, val, intrinsic)


def straddle_price(S, K, T, sigma, r=0.04):
    return bs_price(S, K, T, sigma, r, "call") + bs_price(S, K, T, sigma, r, "put")


def iron_fly_credit(S, center, width, T, sigma, r=0.04):
    """Credit received for selling an iron butterfly: short ATM call+put, long wings at center±width."""
    short_call = bs_price(S, center, T, sigma, r, "call")
    short_put = bs_price(S, center, T, sigma, r, "put")
    long_call = bs_price(S, center + width, T, sigma, r, "call")
    long_put = bs_price(S, center - width, T, sigma, r, "put")
    return short_call + short_put - long_call - long_put


def iron_fly_settlement(S_close, center, width):
    """Intrinsic value owed at expiry on a short iron fly (what the seller pays back)."""
    S_close = np.asarray(S_close)
    short_call = np.maximum(S_close - center, 0.0)
    short_put = np.maximum(center - S_close, 0.0)
    long_call = np.maximum(S_close - (center + width), 0.0)
    long_put = np.maximum((center - width) - S_close, 0.0)
    return short_call + short_put - long_call - long_put


def ratio_credit(S, mid_k, far_k, T, sigma, r=0.04, kind="call"):
    """Credit for selling 2x mid strike + 1x farther strike, same side, no long wing."""
    return 2 * bs_price(S, mid_k, T, sigma, r, kind) + bs_price(S, far_k, T, sigma, r, kind)


def ratio_settlement(S_close, mid_k, far_k, kind="call"):
    """Intrinsic value owed at expiry on the short 2x mid + 1x far ratio position."""
    S_close = np.asarray(S_close)
    if kind == "call":
        return 2 * np.maximum(S_close - mid_k, 0.0) + np.maximum(S_close - far_k, 0.0)
    else:
        return 2 * np.maximum(mid_k - S_close, 0.0) + np.maximum(far_k - S_close, 0.0)


if __name__ == "__main__":
    # sanity checks
    S = 100.0
    c = float(bs_price(S, 100, 1.0, 0.20, r=0.04, kind="call"))
    p = float(bs_price(S, 100, 1.0, 0.20, r=0.04, kind="put"))
    print(f"ATM 1y call @20% vol: {c:.4f} ({c / S:.2%} of spot)  put: {p:.4f}")
    parity_gap = (c - p) - (S - 100 * np.exp(-0.04))
    print(f"put-call parity gap: {parity_gap:.2e}")
    assert abs(parity_gap) < 1e-6, "parity violated"
    assert float(iron_fly_settlement(100.0, 100, 10)) == 0.0
    assert float(iron_fly_settlement(115.0, 100, 10)) == 10.0

    # ratio spread sanity: pin -> 0 owed; beyond mid but before far -> 2x per point;
    # beyond far -> 3x per point (2 mid contracts + 1 far contract all ITM)
    mid_k, far_k = 105, 110
    assert float(ratio_settlement(100.0, mid_k, far_k, "call")) == 0.0
    at_107 = float(ratio_settlement(107.0, mid_k, far_k, "call"))
    at_108 = float(ratio_settlement(108.0, mid_k, far_k, "call"))
    assert abs((at_108 - at_107) - 2.0) < 1e-9, "expected 2x slope between mid and far"
    at_112 = float(ratio_settlement(112.0, mid_k, far_k, "call"))
    at_113 = float(ratio_settlement(113.0, mid_k, far_k, "call"))
    assert abs((at_113 - at_112) - 3.0) < 1e-9, "expected 3x slope beyond far strike"
    print(f"ratio settlement slopes: 2x between mid/far, 3x beyond far strike -- confirmed")
    print("all pricer sanity checks pass")
