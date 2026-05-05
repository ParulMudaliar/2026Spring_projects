"""
test_simulation.py
Unit tests for src/simulation.py

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import numpy as np
import pytest
from typing import Any
from src.simulation import simulate_one_year, run_many
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


def test_stockout_rate_in_range(cfg: dict[str, Any]) -> None:
    """Stockout rate must be between 0 and 1."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert 0.0 <= result["stockout_rate"] <= 1.0


def test_stockout_by_dow_length(cfg: dict[str, Any]) -> None:
    """stockout_by_dow must have exactly 7 entries."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert len(result["stockout_by_dow"]) == 7


def test_stockout_by_hour_length(cfg: dict[str, Any]) -> None:
    """stockout_by_hour must have exactly 24 entries."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert len(result["stockout_by_hour"]) == 24


def test_stockout_by_dow_sums_to_stockout_days(cfg: dict[str, Any]) -> None:
    """Sum of dow stockouts must equal total stockout_days."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert sum(result["stockout_by_dow"]) == result["stockout_days"]


def test_stockout_by_hour_sums_to_stockout_days(cfg: dict[str, Any]) -> None:
    """Sum of hour stockouts must equal total stockout_days."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert sum(result["stockout_by_hour"]) == result["stockout_days"]


def test_dispatch_count_non_negative(cfg: dict[str, Any]) -> None:
    """Dispatch count must be non-negative."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert result["dispatch_count"] >= 0


def test_cash_history_length(cfg: dict[str, Any]) -> None:
    """Cash history must have one entry per simulated day."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert len(result["cash_history"]) == cfg["sim_days"]


def test_cash_never_negative(cfg: dict[str, Any]) -> None:
    """Cash balance must never go below zero."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="fixed", rng=rng)
    assert all(c >= 0 for c in result["cash_history"])


def test_deterministic_with_same_seed(cfg: dict[str, Any]) -> None:
    """Same seed must produce identical results."""
    r1 = simulate_one_year(cfg, rng=np.random.default_rng(99))
    r2 = simulate_one_year(cfg, rng=np.random.default_rng(99))
    assert r1["stockout_rate"] == r2["stockout_rate"]
    assert r1["dispatch_count"] == r2["dispatch_count"]


def test_demand_policy_runs(cfg: dict[str, Any]) -> None:
    """Demand policy must run without error."""
    rng = np.random.default_rng(42)
    result = simulate_one_year(cfg, policy="demand", rng=rng)
    assert 0.0 <= result["stockout_rate"] <= 1.0


def test_high_cash_low_stockout(cfg: dict[str, Any]) -> None:
    """Very high cash should rarely stock out."""
    rich: dict[str, Any] = {
        **cfg, "initial_cash": 10_000_000, "refill_amount": 10_000_000
    }
    rng = np.random.default_rng(42)
    result = simulate_one_year(rich, policy="fixed", rng=rng)
    assert result["stockout_rate"] < 0.1


def test_zero_cash_immediate_stockout(cfg: dict[str, Any]) -> None:
    """Near-zero cash should cause immediate stockouts."""
    broke: dict[str, Any] = {
        **cfg, "initial_cash": 0.01, "refill_amount": 0.01
    }
    rng = np.random.default_rng(42)
    result = simulate_one_year(broke, policy="fixed", rng=rng)
    assert result["stockout_rate"] > 0.0


def test_no_rng_provided(cfg: dict[str, Any]) -> None:
    """Should work without explicit rng (auto-creates one)."""
    result = simulate_one_year(cfg, policy="fixed")
    assert 0.0 <= result["stockout_rate"] <= 1.0


def test_run_many_length(cfg: dict[str, Any]) -> None:
    """run_many must return correct number of results."""
    results = run_many(cfg, policy="fixed", num_runs=3, label="test")
    assert len(results) == 3


def test_run_many_all_have_stockout_rate(cfg: dict[str, Any]) -> None:
    """Every result from run_many must have stockout_rate."""
    results = run_many(cfg, policy="fixed", num_runs=3, label="test")
    assert all("stockout_rate" in r for r in results)


def test_run_many_uses_config_num_runs(cfg: dict[str, Any]) -> None:
    """run_many should use config num_runs if not overridden."""
    cfg2: dict[str, Any] = {**cfg, "num_runs": 2}
    results = run_many(cfg2, policy="fixed", label="test")
    assert len(results) == 2
