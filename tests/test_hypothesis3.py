"""
test_hypothesis3.py
Unit tests for src/h3.py (Hypothesis 3), src/analysis.py, and
src/fit_distributions.py.

Run with:
    pytest tests/test_hypothesis3.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Any
from src.h3 import run_h3, _zero_stockout_result
from src.config import CONFIG


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


# ── H3 tests ────────────────────────────────────────────────────

class TestH3:
    """Tests for Hypothesis 3: peak hour stockout clustering."""

    def test_returns_required_keys(self, cfg: dict[str, Any]) -> None:
        """H3 result must contain all required keys."""
        result = run_h3(cfg)
        for key in ["peak_hours", "peak_stockout_fraction",
                    "expected_peak_fraction", "disproportionality_ratio",
                    "stockouts_by_hour", "stockouts_by_hour_pct",
                    "total_stockouts", "supported", "finding"]:
            assert key in result

    def test_peak_hours_length(self, cfg: dict[str, Any]) -> None:
        """peak_hours must contain exactly 8 hours."""
        result = run_h3(cfg)
        assert len(result["peak_hours"]) == 8

    def test_stockouts_by_hour_length(self, cfg: dict[str, Any]) -> None:
        """stockouts_by_hour must have 24 entries."""
        result = run_h3(cfg)
        assert len(result["stockouts_by_hour"]) == 24

    def test_expected_peak_fraction(self, cfg: dict[str, Any]) -> None:
        """Expected peak fraction must always be 8/24."""
        result = run_h3(cfg)
        assert result["expected_peak_fraction"] == 8 / 24

    def test_peak_stockout_fraction_valid(self, cfg: dict[str, Any]) -> None:
        """Peak stockout fraction must be between 0 and 1."""
        result = run_h3(cfg)
        assert 0.0 <= result["peak_stockout_fraction"] <= 1.0

    def test_supported_is_bool(self, cfg: dict[str, Any]) -> None:
        """supported must be a boolean."""
        result = run_h3(cfg)
        assert isinstance(result["supported"], bool)

    def test_finding_is_string(self, cfg: dict[str, Any]) -> None:
        """finding must be a non-empty string."""
        result = run_h3(cfg)
        assert isinstance(result["finding"], str)
        assert len(result["finding"]) > 0

    def test_zero_stockout_result(self) -> None:
        """Zero stockout fallback must be safe and unsupported."""
        result = _zero_stockout_result([10, 11, 12, 13, 14, 15, 16, 17])
        assert result["total_stockouts"] == 0
        assert result["supported"] is False
        assert len(result["stockouts_by_hour"]) == 24
        assert len(result["peak_hours"]) == 8

    def test_peak_hours_are_sorted(self, cfg: dict[str, Any]) -> None:
        """Peak hours should be returned in sorted order."""
        result = run_h3(cfg)
        assert result["peak_hours"] == sorted(result["peak_hours"])


# ── Analysis plot tests ─────────────────────────────────────────

class TestAnalysis:
    """Tests for analysis.py plotting and export functions."""

    def test_plot_control_run(self) -> None:
        """plot_control_run should run without error."""
        from src.analysis import plot_control_run
        plot_control_run({
            "all_rates": [0.1, 0.12, 0.09, 0.11, 0.10],
            "mean_stockout_rate": 0.104,
        })

    def test_plot_h1(self) -> None:
        """plot_h1 should run without error."""
        from src.analysis import plot_h1
        plot_h1({
            "stockouts_by_dow_pct": [14.0] * 7,
            "weekend_stockout_fraction": 0.286,
            "expected_weekend_fraction": 0.286,
            "disproportionality_ratio": 1.0,
        })

    def test_plot_h2(self) -> None:
        """plot_h2 should run without error."""
        from src.analysis import plot_h2
        plot_h2({
            "fixed": {"all_rates": [0.1, 0.12], "mean_stockout_rate": 0.11},
            "demand": {"all_rates": [0.07, 0.08], "mean_stockout_rate": 0.075},
            "stockout_reduction": 31.8,
            "dispatch_change": 12.5,
        })

    def test_plot_h3(self) -> None:
        """plot_h3 should run without error."""
        from src.analysis import plot_h3
        plot_h3({
            "stockouts_by_hour_pct": [4.0] * 24,
            "peak_hours": [10, 11, 12, 13, 14, 15, 16, 17],
            "peak_stockout_fraction": 0.4,
            "expected_peak_fraction": 0.333,
            "disproportionality_ratio": 1.2,
        })

    def test_save_results(self, tmp_path: Path) -> None:
        """save_results must create results.json."""
        from src import analysis
        # Temporarily redirect outputs dir
        original_outputs = analysis.OUTPUTS
        analysis.OUTPUTS = tmp_path
        try:
            analysis.save_results(
                {"mean_stockout_rate": 0.1, "all_rates": []},
                {"finding": "H1", "supported": True,
                 "weekend_stockout_fraction": 0.4,
                 "weekday_stockout_fraction": 0.6,
                 "expected_weekend_fraction": 0.286,
                 "disproportionality_ratio": 1.4,
                 "stockouts_by_dow": [0] * 7,
                 "stockouts_by_dow_pct": [0.0] * 7,
                 "total_stockouts": 0},
                {"finding": "H2", "supported": True,
                 "stockout_reduction": 30.0, "dispatch_change": 12.0,
                 "fixed": {"mean_stockout_rate": 0.1, "all_rates": [],
                           "std_stockout_rate": 0.02, "p5_stockout_rate": 0.06,
                           "p95_stockout_rate": 0.15, "mean_dispatch_count": 120.0,
                           "std_dispatch_count": 3.0},
                 "demand": {"mean_stockout_rate": 0.07, "all_rates": [],
                            "std_stockout_rate": 0.02, "p5_stockout_rate": 0.04,
                            "p95_stockout_rate": 0.11, "mean_dispatch_count": 135.0,
                            "std_dispatch_count": 5.0}},
                {"finding": "H3", "supported": True,
                 "peak_hours": [10, 11], "peak_stockout_fraction": 0.5,
                 "expected_peak_fraction": 0.333, "disproportionality_ratio": 1.5,
                 "stockouts_by_hour": [0] * 24, "stockouts_by_hour_pct": [0.0] * 24,
                 "total_stockouts": 0},
            )
            assert (tmp_path / "results.json").exists()
        finally:
            analysis.OUTPUTS = original_outputs

    def test_print_summary(self) -> None:
        """print_summary should run without error."""
        from src.analysis import print_summary
        print_summary(
            {"verdict": "PASS", "mean_stockout_rate": 0.1},
            {"finding": "H1 SUPPORTED"},
            {"finding": "H2 SUPPORTED"},
            {"finding": "H3 SUPPORTED"},
        )


# ── fit_distributions.py tests ──────────────────────────────────

class TestFitDistributions:
    """Tests for Phase 1 distribution fitting functions."""

    def test_fit_lognormal_implied_mean(self) -> None:
        """Lognormal mu must produce correct implied mean."""
        from src.fit_distributions import fit_lognormal
        mu, sigma = fit_lognormal(198.13, 0.82)
        implied_mean: float = np.exp(mu + sigma ** 2 / 2)
        assert abs(implied_mean - 198.13) < 0.01

    def test_fit_lognormal_sigma_passthrough(self) -> None:
        """Sigma should be returned unchanged."""
        from src.fit_distributions import fit_lognormal
        _, sigma = fit_lognormal(200.0, 0.5)
        assert sigma == 0.5

    def test_fit_lognormal_different_mean(self) -> None:
        """fit_lognormal should work with any positive mean."""
        from src.fit_distributions import fit_lognormal
        mu, sigma = fit_lognormal(100.0, 0.5)
        implied = np.exp(mu + sigma ** 2 / 2)
        assert abs(implied - 100.0) < 0.01

    def test_fit_poisson_structure(self) -> None:
        """fit_poisson must return 24-element weekday and weekend lists."""
        from src.fit_distributions import fit_poisson
        df: pd.DataFrame = pd.DataFrame({
            "hour": [10, 10, 14, 8, 8],
            "is_weekend": [False, False, True, False, True],
            "atm_id": ["A", "A", "A", "B", "B"],
            "weekday": ["Monday", "Monday", "Saturday", "Tuesday", "Sunday"],
        })
        result = fit_poisson(df)
        assert len(result["weekday"]) == 24
        assert len(result["weekend"]) == 24
        assert all(isinstance(v, float) for v in result["weekday"])

    def test_fit_dow_multipliers_structure(self) -> None:
        """fit_dow_multipliers must return 7 multipliers, Mon=1.0."""
        from src.fit_distributions import fit_dow_multipliers
        df: pd.DataFrame = pd.DataFrame({
            "weekday": ["Monday", "Monday", "Friday", "Friday", "Saturday"],
            "atm_id": ["A", "A", "A", "B", "A"],
        })
        mults: list[float] = fit_dow_multipliers(df)
        assert len(mults) == 7
        assert mults[0] == 1.0  # Monday is baseline
