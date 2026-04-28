import pytest
from src.h3 import run_h3, _zero_stockout_result
from src.config import CONFIG


@pytest.fixture
def cfg():
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
def test_returns_required_keys(cfg):
    """H3 result must contain all required keys."""
    result = run_h3(cfg)
    for key in [
        "peak_hours",
        "peak_stockout_fraction",
        "expected_peak_fraction",
        "disproportionality_ratio",
        "stockouts_by_hour",
        "stockouts_by_hour_pct",
        "total_stockouts",
        "supported",
        "finding",
    ]:
        assert key in result


def test_peak_hours_length(cfg):
    """peak_hours must always contain exactly 8 hours."""
    result = run_h3(cfg)
    assert len(result["peak_hours"]) == 8


def test_stockouts_by_hour_length(cfg):
    """stockouts_by_hour must have 24 entries."""
    result = run_h3(cfg)
    assert len(result["stockouts_by_hour"]) == 24

def test_expected_peak_fraction(cfg):
    """Expected peak fraction must always be 8/24."""
    result = run_h3(cfg)
    assert result["expected_peak_fraction"] == 8 / 24


def test_peak_stockout_fraction_valid(cfg):
    """Peak stockout fraction must be between 0 and 1."""
    result = run_h3(cfg)
    assert 0.0 <= result["peak_stockout_fraction"] <= 1.0


def test_supported_is_bool(cfg):
    """supported field must be a boolean."""
    result = run_h3(cfg)
    assert isinstance(result["supported"], bool)