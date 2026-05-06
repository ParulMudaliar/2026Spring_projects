"""
test_hypothesis2.py
Unit tests for src/h2.py

Run with:
    pytest tests/test_hypothesis2.py -v
"""

import pytest
from typing import Any
from src.h2 import run_h2, _summarize
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


def test_returns_required_keys(cfg: dict[str, Any]) -> None:
    """H2 result must contain all required keys."""
    result = run_h2(cfg)
    for key in [
        "fixed", "demand",
        "stockout_reduction", "dispatch_change",
        "supported", "finding"
    ]:
        assert key in result


def test_stockout_rates_valid(cfg: dict[str, Any]) -> None:
    """Both policy stockout rates must be between 0 and 1."""
    result = run_h2(cfg)
    assert 0.0 <= result["fixed"]["mean_stockout_rate"] <= 1.0
    assert 0.0 <= result["demand"]["mean_stockout_rate"] <= 1.0


def test_dispatch_counts_positive(cfg: dict[str, Any]) -> None:
    """Both dispatch counts must be non-negative."""
    result = run_h2(cfg)
    assert result["fixed"]["mean_dispatch_count"] >= 0
    assert result["demand"]["mean_dispatch_count"] >= 0


def test_supported_is_bool(cfg: dict[str, Any]) -> None:
    """supported field must be a boolean."""
    result = run_h2(cfg)
    assert isinstance(result["supported"], bool)


def test_finding_is_string(cfg: dict[str, Any]) -> None:
    """finding field must be a non-empty string."""
    result = run_h2(cfg)
    assert isinstance(result["finding"], str)
    assert len(result["finding"]) > 0


def test_fixed_stats_has_all_keys(cfg: dict[str, Any]) -> None:
    """fixed stats dict must have summary statistics."""
    result = run_h2(cfg)
    for key in ["mean_stockout_rate", "std_stockout_rate",
                "p5_stockout_rate", "p95_stockout_rate",
                "mean_dispatch_count", "std_dispatch_count", "all_rates"]:
        assert key in result["fixed"]


def test_demand_stats_has_all_keys(cfg: dict[str, Any]) -> None:
    """demand stats dict must have summary statistics."""
    result = run_h2(cfg)
    for key in ["mean_stockout_rate", "std_stockout_rate",
                "p5_stockout_rate", "p95_stockout_rate",
                "mean_dispatch_count", "std_dispatch_count", "all_rates"]:
        assert key in result["demand"]


def test_summarize_basic() -> None:
    """_summarize must compute correct mean from synthetic results."""
    results: list[dict[str, Any]] = [
        {"stockout_rate": 0.1, "dispatch_count": 100,
         "stockout_by_dow": [0] * 7, "stockout_by_hour": [0] * 24,
         "stockout_days": 10, "cash_history": []},
        {"stockout_rate": 0.2, "dispatch_count": 110,
         "stockout_by_dow": [0] * 7, "stockout_by_hour": [0] * 24,
         "stockout_days": 20, "cash_history": []},
    ]
    s = _summarize(results)
    assert abs(s["mean_stockout_rate"] - 0.15) < 0.001
    assert abs(s["mean_dispatch_count"] - 105.0) < 0.001
    assert len(s["all_rates"]) == 2


def test_summarize_single_run() -> None:
    """_summarize should handle a single result."""
    results: list[dict[str, Any]] = [
        {"stockout_rate": 0.05, "dispatch_count": 120,
         "stockout_by_dow": [0] * 7, "stockout_by_hour": [0] * 24,
         "stockout_days": 5, "cash_history": []},
    ]
    s = _summarize(results)
    assert s["mean_stockout_rate"] == 0.05
    assert s["std_stockout_rate"] == 0.0
