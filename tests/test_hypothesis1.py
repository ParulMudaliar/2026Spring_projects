"""
test_hypothesis1.py
Unit tests for src/h1.py (Hypothesis 1) and src/phase2_control.py (Phase 2).

Run with:
    pytest tests/test_hypothesis1.py -v
"""

import numpy as np
import pytest
from typing import Any
from src.config import CONFIG, get_param


@pytest.fixture
def cfg() -> dict[str, Any]:
    """Minimal fast config for testing."""
    return {
        **CONFIG,
        "sim_days": 30,
        "num_runs": 5,
        "lognormal_mu": 4.61,
        "lognormal_sigma": 0.82,
        "poisson_lambda_weekday": [0.5] * 24,
        "poisson_lambda_weekend": [0.8] * 24,
        "dow_multipliers": [1.0] * 7,
        "benchmark_min": 0.0,
        "benchmark_max": 1.0,
    }


# ── config.py tests ─────────────────────────────────────────────

class TestConfig:
    """Tests for the config module and get_param fallback logic."""

    def test_get_param_returns_fitted(self) -> None:
        """When fitted value exists, return it directly."""
        assert get_param(CONFIG, "lognormal_sigma") == 0.82

    def test_get_param_returns_fallback_when_none(self) -> None:
        """When fitted value is None, return the fallback."""
        cfg: dict[str, Any] = {**CONFIG, "lognormal_mu": None}
        assert get_param(cfg, "lognormal_mu") == CONFIG["fallback_lognormal_mu"]

    def test_get_param_poisson_weekday_fallback(self) -> None:
        """Poisson weekday fallback mapping works."""
        cfg: dict[str, Any] = {**CONFIG, "poisson_lambda_weekday": None}
        result = get_param(cfg, "poisson_lambda_weekday")
        assert result == CONFIG["fallback_poisson_weekday"]

    def test_get_param_poisson_weekend_fallback(self) -> None:
        """Poisson weekend fallback mapping works."""
        cfg: dict[str, Any] = {**CONFIG, "poisson_lambda_weekend": None}
        result = get_param(cfg, "poisson_lambda_weekend")
        assert result == CONFIG["fallback_poisson_weekend"]

    def test_get_param_dow_fallback(self) -> None:
        """Day-of-week multiplier fallback works."""
        cfg: dict[str, Any] = {**CONFIG, "dow_multipliers": None}
        result = get_param(cfg, "dow_multipliers")
        assert result == CONFIG["fallback_dow_multipliers"]

    def test_get_param_unknown_key(self) -> None:
        """Unknown keys return None."""
        assert get_param(CONFIG, "nonexistent_key_xyz") is None

    def test_config_has_required_keys(self) -> None:
        """CONFIG dict must contain all required simulation parameters."""
        required: list[str] = [
            "initial_cash", "refill_amount", "fixed_refill_days",
            "demand_threshold", "weibull_shape", "weibull_scale",
            "sim_days", "num_runs", "benchmark_min", "benchmark_max",
        ]
        for key in required:
            assert key in CONFIG


# ── Phase 2 control run tests ───────────────────────────────────

class TestPhase2Control:
    """Tests for the Phase 2 control run validation."""

    def test_run_control_returns_required_keys(self, cfg: dict[str, Any]) -> None:
        """run_control result must contain all expected keys."""
        from src.phase2_control import run_control
        result = run_control(cfg)
        for key in ["mean_stockout_rate", "std_stockout_rate",
                     "min_stockout_rate", "max_stockout_rate",
                     "mean_dispatch_count", "benchmark_pass",
                     "verdict", "all_rates"]:
            assert key in result

    def test_run_control_passes_wide_benchmark(self, cfg: dict[str, Any]) -> None:
        """With wide benchmark bounds, control should pass."""
        from src.phase2_control import run_control
        result = run_control(cfg)
        assert result["benchmark_pass"] is True
        assert "PASS" in result["verdict"]

    def test_run_control_fails_narrow_benchmark(self, cfg: dict[str, Any]) -> None:
        """With impossibly narrow bounds, control should fail."""
        from src.phase2_control import run_control
        narrow: dict[str, Any] = {
            **cfg, "benchmark_min": 0.99, "benchmark_max": 1.0
        }
        result = run_control(narrow)
        assert result["benchmark_pass"] is False
        assert "WARNING" in result["verdict"]

    def test_all_rates_length_matches_num_runs(self, cfg: dict[str, Any]) -> None:
        """all_rates list must have exactly num_runs entries."""
        from src.phase2_control import run_control
        result = run_control(cfg)
        assert len(result["all_rates"]) == cfg["num_runs"]

    def test_print_results_runs(self) -> None:
        """_print_results must run without error."""
        from src.phase2_control import _print_results
        _print_results(0.10, 0.02, 0.05, 0.18, 122.0, "PASS")


# ── H1 tests ────────────────────────────────────────────────────

class TestH1:
    """Tests for Hypothesis 1: weekend stockout disproportionality."""

    def test_returns_required_keys(self, cfg: dict[str, Any]) -> None:
        """H1 result must contain all expected keys."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        for key in ["weekend_stockout_fraction", "weekday_stockout_fraction",
                     "expected_weekend_fraction", "disproportionality_ratio",
                     "stockouts_by_dow", "stockouts_by_dow_pct",
                     "total_stockouts", "supported", "finding"]:
            assert key in result

    def test_expected_weekend_fraction(self, cfg: dict[str, Any]) -> None:
        """Expected weekend fraction must always be 2/7."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        assert result["expected_weekend_fraction"] == 2 / 7

    def test_supported_is_bool(self, cfg: dict[str, Any]) -> None:
        """supported must be a boolean."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        assert isinstance(result["supported"], bool)

    def test_stockouts_by_dow_length(self, cfg: dict[str, Any]) -> None:
        """stockouts_by_dow must have 7 entries."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        assert len(result["stockouts_by_dow"]) == 7

    def test_stockouts_by_dow_pct_length(self, cfg: dict[str, Any]) -> None:
        """stockouts_by_dow_pct must have 7 entries."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        assert len(result["stockouts_by_dow_pct"]) == 7

    def test_fractions_sum_to_one(self, cfg: dict[str, Any]) -> None:
        """Weekend + weekday fractions must sum to 1.0."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        if result["total_stockouts"] > 0:
            total = (result["weekend_stockout_fraction"] +
                     result["weekday_stockout_fraction"])
            assert abs(total - 1.0) < 0.001

    def test_zero_stockout_result(self) -> None:
        """Zero stockout fallback must be safe and unsupported."""
        from src.h1 import _zero_stockout_result
        result = _zero_stockout_result()
        assert result["total_stockouts"] == 0
        assert result["supported"] is False
        assert len(result["stockouts_by_dow"]) == 7

    def test_finding_is_nonempty_string(self, cfg: dict[str, Any]) -> None:
        """finding field must be a non-empty string."""
        from src.h1 import run_h1
        result = run_h1(cfg)
        assert isinstance(result["finding"], str)
        assert len(result["finding"]) > 0
