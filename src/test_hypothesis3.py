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