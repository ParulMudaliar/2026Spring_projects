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
    """Simulate one ATM for 365 days under a replenishment policy.

    Each day:
      1. Check if pending truck arrives
      2. For each hour: draw Poisson arrivals, draw Lognormal
         withdrawals, deplete cash, record stockouts
      3. Decide whether to dispatch a truck

    Parameters
    ----------
    config : dict
        Configuration from config.py. Uses fitted params if available,
        otherwise uses fallback params automatically.
    policy : str
        'fixed'  — refill every fixed_refill_days regardless of cash.
        'demand' — refill when cash drops below demand_threshold.
    rng : np.random.Generator, optional
        Seeded generator for reproducibility. Created fresh if None.

    Returns
    -------
    dict with keys:
        stockout_days    : int
        stockout_rate    : float
        stockout_by_dow  : list[int]  — 7 values, Mon=0 to Sun=6
        stockout_by_hour : list[int]  — 24 values, hour 0 to 23
        dispatch_count   : int
        cash_history     : list[float]

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import CONFIG
    >>> rng = np.random.default_rng(42)
    >>> result = simulate_one_year(CONFIG, policy='fixed', rng=rng)
    >>> 0.0 <= result['stockout_rate'] <= 1.0
    True
    >>> len(result['stockout_by_dow']) == 7
    True
    >>> len(result['stockout_by_hour']) == 24
    True
    >>> result['dispatch_count'] >= 0
    True
    """
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
    """Run simulate_one_year num_runs times and return all results.

    Parameters
    ----------
    config : dict
        Simulation configuration.
    policy : str
        'fixed' or 'demand'.
    num_runs : int, optional
        Overrides config['num_runs'] if given.
    base_seed : int
        Run i uses seed base_seed + i for full reproducibility.
    label : str
        Progress label printed to terminal.

    Returns
    -------
    list[dict]
        One result dict per run.

    Examples
    --------
    >>> from src.config import CONFIG
    >>> results = run_many(CONFIG, policy='fixed', num_runs=3, label='test')
    >>> len(results) == 3
    True
    >>> all('stockout_rate' in r for r in results)
    True
    """
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