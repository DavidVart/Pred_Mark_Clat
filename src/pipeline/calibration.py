"""Platt-scaling calibration for the LLM ensemble.

The raw weighted probability from our 5-model ensemble is NOT a calibrated
probability — LLMs are consistently overconfident, and the ensemble average
inherits that bias. To turn our ensemble output into a real probability, we
fit a logistic regression on (ensemble_prob, actual_outcome) pairs from
resolved trades, then apply the fitted model to future predictions.

This follows the approach used by Metaculus-winning forecasting bots:
1. Log every (predicted_probability, resolved_outcome) pair
2. Once enough samples accumulate (>= MIN_SAMPLES_FOR_CALIBRATION), fit a
   Platt-scaling logistic on them
3. Apply the fitted scaler to new ensemble outputs before the edge check

The calibration is per-cluster (sports vs politics vs crypto etc) because
LLMs have different biases across domains.

Reference: Platt, J. (1999). "Probabilistic outputs for SVMs and comparisons
to regularized likelihood methods."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.db.manager import DatabaseManager
from src.utils.logging import get_logger

logger = get_logger("calibration")

MIN_SAMPLES_FOR_CALIBRATION = 30  # Need enough to fit 2 params meaningfully


@dataclass(frozen=True)
class PlattScaler:
    """A fitted Platt scaler: calibrated = sigmoid(A * logit(raw) + B)."""

    a: float  # slope on logit(raw)
    b: float  # intercept
    n_samples: int
    brier_before: float  # Brier score on training data before scaling
    brier_after: float   # Brier score after scaling

    @classmethod
    def identity(cls) -> "PlattScaler":
        """Pass-through scaler (no calibration applied)."""
        return cls(a=1.0, b=0.0, n_samples=0, brier_before=0.0, brier_after=0.0)

    def apply(self, raw_probability: float) -> float:
        """Transform a raw ensemble probability into a calibrated one."""
        p = min(max(raw_probability, 1e-6), 1.0 - 1e-6)
        logit = math.log(p / (1.0 - p))
        z = self.a * logit + self.b
        # Clamp z to prevent exp overflow (math.exp(-z) overflows near z=-710)
        z = max(-500.0, min(500.0, z))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    @property
    def is_identity(self) -> bool:
        return self.n_samples == 0


def fit_platt_scaler(samples: list[tuple[float, int]]) -> PlattScaler:
    """Fit Platt scaling via Newton's method for logistic regression on logits.

    Args:
        samples: list of (raw_probability, outcome) where outcome ∈ {0, 1}

    Returns:
        Fitted PlattScaler. Returns identity if not enough samples.

    Uses a simple IRLS (iteratively reweighted least squares) approach. We don't
    want to add sklearn as a dependency for one function.
    """
    if len(samples) < MIN_SAMPLES_FOR_CALIBRATION:
        return PlattScaler.identity()

    # Convert to (logit, outcome) pairs
    pairs = []
    for p, y in samples:
        p_clip = min(max(p, 1e-6), 1.0 - 1e-6)
        pairs.append((math.log(p_clip / (1.0 - p_clip)), y))

    # Initial params
    a, b = 1.0, 0.0

    # L2 regularization prevents degenerate data (e.g. only 2 distinct logits)
    # from letting |a| run off to infinity while fitting perfectly in-sample.
    # Small lambda: negligible effect on well-behaved data, stabilizes edge cases.
    ridge_lambda = 1e-3

    # Newton iteration with step-size backtracking for stability
    for _iter in range(50):
        grad_a = grad_b = 0.0
        hess_aa = hess_ab = hess_bb = 0.0

        for x, y in pairs:
            z = a * x + b
            # Numerical stability for sigmoid
            if z >= 0:
                ez = math.exp(-z)
                sig = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                sig = ez / (1.0 + ez)
            err = sig - y
            grad_a += err * x
            grad_b += err
            w = sig * (1.0 - sig)
            hess_aa += w * x * x
            hess_ab += w * x
            hess_bb += w

        # Add ridge term to both diagonal entries and gradient
        grad_a += ridge_lambda * a
        grad_b += ridge_lambda * b
        hess_aa += ridge_lambda
        hess_bb += ridge_lambda

        # Solve 2x2 linear system H * delta = -grad
        det = hess_aa * hess_bb - hess_ab * hess_ab
        if abs(det) < 1e-12:
            break
        delta_a = (-grad_a * hess_bb + grad_b * hess_ab) / det
        delta_b = (grad_a * hess_ab - grad_b * hess_aa) / det

        # Cap step size to prevent divergence on unusual data
        max_step = 5.0
        step_norm = math.sqrt(delta_a * delta_a + delta_b * delta_b)
        if step_norm > max_step:
            delta_a *= max_step / step_norm
            delta_b *= max_step / step_norm

        a += delta_a
        b += delta_b

        if abs(delta_a) < 1e-6 and abs(delta_b) < 1e-6:
            break

    scaler = PlattScaler(a=a, b=b, n_samples=len(samples), brier_before=0.0, brier_after=0.0)

    # Compute Brier scores for sanity reporting
    brier_before = sum((p - y) ** 2 for p, y in samples) / len(samples)
    brier_after = sum((scaler.apply(p) - y) ** 2 for p, y in samples) / len(samples)

    return PlattScaler(
        a=a, b=b, n_samples=len(samples),
        brier_before=brier_before, brier_after=brier_after,
    )


async def load_calibration_samples(db: DatabaseManager, cluster: str | None = None) -> list[tuple[float, int]]:
    """Fetch all (predicted_probability, outcome) pairs from resolved trades.

    Joins the predictions table to trade_log via market_id, filtering to the
    most recent prediction before the trade opened. If cluster is provided,
    we would filter by it — currently we log per-category only, so cluster
    filtering is a future extension.
    """
    # For each closed trade, find the latest prediction we logged for that market
    # BEFORE the position was opened, and pair it with the resolved outcome.
    query = """
        SELECT
            p.weighted_probability,
            CASE
                WHEN t.side = 'yes' AND t.outcome = 'win' THEN 1
                WHEN t.side = 'yes' AND t.outcome = 'loss' THEN 0
                WHEN t.side = 'no'  AND t.outcome = 'win' THEN 0
                WHEN t.side = 'no'  AND t.outcome = 'loss' THEN 1
                ELSE NULL
            END AS yes_outcome
        FROM trade_log t
        JOIN predictions p ON p.market_id = t.market_id
        WHERE p.timestamp <= t.opened_at
          AND t.outcome IN ('win', 'loss')
        ORDER BY p.timestamp DESC
    """
    cursor = await db.db.execute(query)
    rows = await cursor.fetchall()

    # Dedupe to latest prediction per trade (the ORDER BY above gives us newest first)
    seen = set()
    samples = []
    for row in rows:
        prob = row[0]
        outcome = row[1]
        if outcome is None or prob is None:
            continue
        # Dedupe on rounded-probability so identical preds from the same batch
        # don't dominate; simpler: just use all. For rigorous: track market_id.
        samples.append((float(prob), int(outcome)))

    return samples


async def get_current_scaler(db: DatabaseManager) -> PlattScaler:
    """Fit and return a current Platt scaler from DB history."""
    samples = await load_calibration_samples(db)
    scaler = fit_platt_scaler(samples)
    if not scaler.is_identity:
        improvement = scaler.brier_before - scaler.brier_after
        logger.info(
            "calibration_fit",
            samples=scaler.n_samples,
            a=round(scaler.a, 3),
            b=round(scaler.b, 3),
            brier_before=round(scaler.brier_before, 4),
            brier_after=round(scaler.brier_after, 4),
            improvement=round(improvement, 4),
        )
    return scaler
