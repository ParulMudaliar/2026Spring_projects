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

    # Pull parameters— uses fallback if fitted values are None
    mu =get_param(config, "lognormal_mu")
    sigma = get_param(config, "lognormal_sigma")
    weekday_lambda = np.array(get_param(config, "poisson_lambda_weekday"), dtype=float)
    weekend_lambda = np.array(get_param(config, "poisson_lambda_weekend"), dtype=float)
    dow_multipliers = np.array(get_param(config, "dow_multipliers"), dtype=float)

    cash= float(config["initial_cash"])
    refill_amount = float(config["refill_amount"])
    fixed_refill_days =config["fixed_refill_days"]
    demand_threshold = float(config["demand_threshold"])
    weibull_shape =config["weibull_shape"]
    weibull_scale = config["weibull_scale"]
    sim_days = config["sim_days"]

    # Tracking
    stockout_days = 0
    stockout_by_dow = [0] * 7
    stockout_by_hour = [0] * 24
    dispatch_count = 0
    cash_history = []
    days_since_dispatch = 0
    pending_refill_in = None

    for day in range(sim_days):
        dow = day % 7 # 0=Mon, 6=Sun
        is_weekend = dow >= 5
        hourly_lambda = weekend_lambda if is_weekend else weekday_lambda
        multiplier = dow_multipliers[dow]

        # Truck arrives today
        if pending_refill_in is not None:
            pending_refill_in -= 1
            if pending_refill_in <= 0:
                cash += refill_amount
                pending_refill_in = None
                days_since_dispatch = 0

        day_had_stockout = False
        stockout_hour = None

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
                    day_had_stockout = True
                    stockout_hour = hour
                    break

            if day_had_stockout:
                break

        if day_had_stockout:
            stockout_days += 1
            stockout_by_dow[dow] += 1
            if stockout_hour is not None:
                stockout_by_hour[stockout_hour] += 1

        cash_history.append(round(cash, 2))
        days_since_dispatch += 1

        # Dispatch decision
        if pending_refill_in is None:
            dispatch = False
            if policy == "fixed" and days_since_dispatch >= fixed_refill_days:
                dispatch = True
            elif policy == "demand" and cash < demand_threshold:
                dispatch = True

            if dispatch:
                delay = float(rng.weibull(weibull_shape) * weibull_scale)
                pending_refill_in = max(1, round(delay))
                dispatch_count += 1

    return {
        "stockout_days": stockout_days,
        "stockout_rate": round(stockout_days / sim_days, 6),
        "stockout_by_dow": stockout_by_dow,
        "stockout_by_hour": stockout_by_hour,
        "dispatch_count": dispatch_count,
        "cash_history": cash_history,
    }

def run_many(
    config: dict,
    policy: str = "fixed",
    num_runs: int | None = None,
    base_seed: int = 0,
    label: str = "",
) -> list[dict]:
    n = num_runs if num_runs is not None else config["num_runs"]
    results = []
    print_every = max(1, n // 5)

    for i in range(n):
        if i % print_every == 0:
            pct = int(i / n * 100)
            print(f"  {label}: {pct}% ({i}/{n})")
        rng = np.random.default_rng(base_seed + i)
        results.append(simulate_one_year(config, policy=policy, rng=rng))

    print(f"  {label}: 100% ({n}/{n}) done")
    return results