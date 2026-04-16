"""Tests for Platt scaling calibration layer."""

from __future__ import annotations

import math
import random

from src.pipeline.calibration import (
    MIN_SAMPLES_FOR_CALIBRATION,
    PlattScaler,
    fit_platt_scaler,
)


class TestIdentityScaler:
    def test_identity_passes_through(self):
        s = PlattScaler.identity()
        assert abs(s.apply(0.5) - 0.5) < 1e-9
        assert abs(s.apply(0.3) - 0.3) < 1e-6
        assert abs(s.apply(0.9) - 0.9) < 1e-6

    def test_identity_is_flagged(self):
        assert PlattScaler.identity().is_identity


class TestPlattFitting:
    def test_too_few_samples_returns_identity(self):
        samples = [(0.5, 1), (0.6, 0), (0.4, 1)]
        s = fit_platt_scaler(samples)
        assert s.is_identity

    def test_well_calibrated_data_fits_near_identity(self):
        """If raw probs already match outcomes, scaler should be near identity."""
        random.seed(42)
        samples = []
        for _ in range(500):
            p = random.random()
            y = 1 if random.random() < p else 0  # outcomes match probabilities
            samples.append((p, y))
        s = fit_platt_scaler(samples)
        # Scaler shouldn't drastically reshape well-calibrated data
        assert abs(s.apply(0.5) - 0.5) < 0.15
        assert s.n_samples == 500

    def test_overconfident_model_gets_pulled_toward_half(self):
        """If raw probs are too extreme, the scaler should moderate them."""
        random.seed(42)
        samples = []
        # The "real" probability of any outcome is 60% regardless of raw pred
        for _ in range(500):
            raw = 0.1 if random.random() < 0.5 else 0.9  # model screams 10% or 90%
            y = 1 if random.random() < 0.60 else 0  # but reality is 60/40
            samples.append((raw, y))
        s = fit_platt_scaler(samples)
        # After calibration, 0.9 should map to something closer to ~0.60
        calibrated_high = s.apply(0.9)
        calibrated_low = s.apply(0.1)
        # Should be pulled toward the actual base rate of 60%
        assert calibrated_high < 0.9
        assert calibrated_low > 0.1

    def test_brier_improves_when_scaler_helps(self):
        """Brier after fitting should be <= Brier before on training data."""
        random.seed(123)
        samples = []
        for _ in range(200):
            raw = random.random()
            # Introduce a systematic bias: add 0.1 to raw before comparing
            biased_truth = min(1.0, max(0.0, raw - 0.1))
            y = 1 if random.random() < biased_truth else 0
            samples.append((raw, y))
        s = fit_platt_scaler(samples)
        assert s.brier_after <= s.brier_before + 1e-9

    def test_apply_clamps_extremes(self):
        s = fit_platt_scaler([(0.5, 1)] * MIN_SAMPLES_FOR_CALIBRATION)
        # Should not raise math domain errors on 0 or 1
        result_zero = s.apply(0.0)
        result_one = s.apply(1.0)
        assert 0.0 <= result_zero <= 1.0
        assert 0.0 <= result_one <= 1.0

    def test_monotonic(self):
        """A calibrated mapping should be monotonic: higher raw -> higher cal."""
        random.seed(7)
        samples = [(random.random(), random.randint(0, 1)) for _ in range(200)]
        s = fit_platt_scaler(samples)
        last = -1.0
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cal = s.apply(p)
            # Monotonic if a > 0 (typical). We allow a small tolerance.
            if s.a > 0:
                assert cal >= last - 1e-9
            last = cal
