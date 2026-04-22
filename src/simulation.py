"""
simulation.py
Phase 1 — Single ATM simulation core loop.

This is the shared engine used by Phase 2 control run and all
three hypothesis experiments. Every other file imports this.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""
import numpy as np
from src.config import CONFIG, get_param


def simulate_one_year(config: dict, policy: str = "fixed", rng: np.random.Generator | None = None,) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    # Pull parameters
    mu = get_param(config, "lognormal_mu")
    sigma = get_param(config, "lognormal_sigma")
    weekday_lambda = np.array(get_param(config, "poisson_lambda_weekday"), dtype=float)
    weekend_lambda = np.array(get_param(config, "poisson_lambda_weekend"), dtype=float)
    dow_multipliers = np.array(get_param(config, "dow_multipliers"), dtype=float)

    cash = float(config["initial_cash"])
    sim_days = config["sim_days"]

    # Tracking
    stockout_days = 0

    for day in range(sim_days):
        dow = day % 7
        is_weekend = dow >= 5
        hourly_lambda = weekend_lambda if is_weekend else weekday_lambda
        multiplier = dow_multipliers[dow]

        for hour in range(24):
            lam = float(hourly_lambda[hour] * multiplier)
            if lam <= 0:
                continue

            n_arrivals = int(rng.poisson(lam))
            if n_arrivals == 0:
                continue

            withdrawals = rng.lognormal(mu, sigma, size=n_arrivals)

            for w in withdrawals:
                if cash >= w:
                    cash -= w
                else:
                    stockout_days += 1
                    break

    return {
        "stockout_days": stockout_days,
    }