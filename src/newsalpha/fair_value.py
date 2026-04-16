"""Fair-value computation for short-duration binary "price-higher" markets.

For a market of the form "Will BTC close above $X at time T?", the fair
probability depends on:
    - Current BTC spot price (now)
    - Strike X (usually = BTC price at window open for "up/down" markets)
    - Time remaining until T
    - BTC short-term log-return volatility

Assuming log-returns are ~N(0, σ²·t) (standard geometric Brownian motion over
short horizons, no drift on 5-min timescales), the fair probability of the
price being above strike at time T is:

    P(S_T > X | S_t) = Φ( ln(S_t / X) / (σ · sqrt(Δt)) )

where Φ is the standard normal CDF and Δt is the time remaining as a fraction
of the volatility horizon.

We use realized 5-minute BTC volatility as a baseline (typical σ_5min ≈ 0.3%,
which annualizes to ~60% vol). For very short windows (<30s remaining), this
model degenerates — in that regime we clamp the probability.

We deliberately keep the math simple: drift = 0, no jumps, no dividends. For
5-minute BTC markets, those assumptions are essentially correct; for longer
windows you'd want a richer model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Realized 5-minute BTC volatility, sampled from historical. Will be overridden
# at runtime if we attach a live volatility estimator; this is the prior.
DEFAULT_SIGMA_5MIN = 0.003  # 0.3% std-dev of log-return over 5 minutes


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf — no scipy needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class FairValueParams:
    sigma_5min: float = DEFAULT_SIGMA_5MIN
    # Clamp fair values within this band to avoid division-by-near-zero noise
    clamp_min: float = 0.01
    clamp_max: float = 0.99
    # Minimum seconds remaining before we refuse to quote (pure time-decay zone)
    min_seconds_remaining: float = 5.0


def fair_yes_probability(
    spot: float,
    strike: float,
    seconds_remaining: float,
    window_total_seconds: float = 300.0,
    params: FairValueParams | None = None,
) -> float:
    """Fair probability the market resolves YES ("price > strike") at resolution time.

    Args:
        spot: current BTC spot price
        strike: the price the market compares against (typically the opening price)
        seconds_remaining: time left until resolution
        window_total_seconds: total duration of the market window (e.g. 300 for 5M)
        params: volatility config

    Returns:
        probability in [clamp_min, clamp_max]
    """
    p = params or FairValueParams()

    if spot <= 0 or strike <= 0:
        return 0.5

    if seconds_remaining <= p.min_seconds_remaining:
        # Inside the last N seconds: probability is essentially just "is spot > strike"
        # clamped slightly away from 0/1 so trades aren't triggered by rounding.
        if spot > strike:
            return p.clamp_max
        elif spot < strike:
            return p.clamp_min
        else:
            return 0.5

    # Scale sigma to the remaining horizon via Brownian motion scaling.
    # sigma_5min is the std-dev of log-return over 5 min (300 seconds).
    # For Δt seconds remaining, σ(Δt) = σ_5min * sqrt(Δt / 300).
    # NOTE: window_total_seconds is intentionally NOT used here — sigma scales
    # purely with time elapsed, independent of what window length the market is.
    sigma_remaining = p.sigma_5min * math.sqrt(seconds_remaining / 300.0)

    if sigma_remaining <= 0:
        return 0.5

    log_moneyness = math.log(spot / strike)
    z = log_moneyness / sigma_remaining
    prob = _normal_cdf(z)

    # Clamp
    return max(p.clamp_min, min(p.clamp_max, prob))


def fair_no_probability(
    spot: float,
    strike: float,
    seconds_remaining: float,
    window_total_seconds: float = 300.0,
    params: FairValueParams | None = None,
) -> float:
    return 1.0 - fair_yes_probability(spot, strike, seconds_remaining, window_total_seconds, params)
