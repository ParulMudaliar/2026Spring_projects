"""
simulation.py
Phase 1 — Single ATM simulation core loop.

This is the shared engine used by Phase 2 control run and all
three hypothesis experiments. Every other file imports this.

Uses Numba JIT compilation for the inner hourly cash-depletion
loop to accelerate the 10,000-run Monte Carlo across 365 days.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

import numpy as np
from numba import njit
from src.config import CONFIG, get_param
from typing import Any

@njit(cache=True)
def _deplete_cash_hourly(
    cash: float,
    hourly_lambda: np.ndarray,
    multiplier: float,
    mu: float,
    sigma: float,
    rng_poisson_samples: np.ndarray,
    rng_lognormal_samples: list[np.ndarray],
) -> tuple[float, bool, int]:
    """JIT-compiled inner loop: deplete cash through 24 hours.

    Processes pre-drawn random samples to determine if a stockout
    occurs and at which hour.

    Parameters
    ----------
    cash : float
        Current cash balance at start of day.
    hourly_lambda : np.ndarray
        24-element array of Poisson lambda values for this day type.
    multiplier : float
        Day-of-week demand multiplier.
    mu : float
        Lognormal mu parameter (unused here — samples pre-drawn).
    sigma : float
        Lognormal sigma parameter (unused here — samples pre-drawn).
    rng_poisson_samples : np.ndarray
        Pre-drawn Poisson arrival counts, shape (24,).
    rng_lognormal_samples : list of np.ndarray
        Pre-drawn lognormal withdrawal amounts per hour.

    Returns
    -------
    tuple[float, bool, int]
        (remaining_cash, had_stockout, stockout_hour)

    Note
    ----
    Numba JIT-compiled — cannot be called directly in doctests.
    Tested indirectly via simulate_one_year doctests.
    """
    had_stockout = False
    stockout_hour = -1

    for hour in range(24):
        n_arrivals = rng_poisson_samples[hour]
        if n_arrivals == 0:
            continue
        withdrawals = rng_lognormal_samples[hour]
        for j in range(n_arrivals):
            w = withdrawals[j]
            if cash >= w:
                cash -= w
            else:
                had_stockout = True
                stockout_hour = hour
                break
        if had_stockout:
            break

    return cash, had_stockout, stockout_hour


def simulate_one_year(
    config: dict[str, Any],
    policy: str = "fixed",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Simulate one ATM for 365 days under a replenishment policy.
    Each day:
      1. Check if pending truck arrives
      2. For each hour: draw Poisson arrivals, draw Lognormal
         withdrawals, deplete cash, record stockouts
      3. Decide whether to dispatch a truck

    Parameters
    ----------
    config : dict[str, Any]
        Configuration from config.py.
    policy : str
        'fixed' or 'demand'.
    rng : np.random.Generator or None
        Seeded generator for reproducibility.

    Returns
    -------
    dict[str, Any]
        Keys: stockout_days, stockout_rate, stockout_by_dow,
        stockout_by_hour, dispatch_count, cash_history.

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import CONFIG
    >>> cfg = {**CONFIG, 'sim_days': 30, 'num_runs': 3}
    >>> rng = np.random.default_rng(42)
    >>> result = simulate_one_year(cfg, policy='fixed', rng=rng)
    >>> 0.0 <= result['stockout_rate'] <= 1.0
    True
    >>> len(result['stockout_by_dow']) == 7
    True
    >>> len(result['stockout_by_hour']) == 24
    True
    >>> result['dispatch_count'] >= 0
    True
    >>> len(result['cash_history']) == 30
    True

    >>> result2 = simulate_one_year(cfg, policy='demand', rng=np.random.default_rng(42))
    >>> 0.0 <= result2['stockout_rate'] <= 1.0
    True

    >>> rich = {**cfg, 'initial_cash': 99_999_999, 'refill_amount': 99_999_999}
    >>> r = simulate_one_year(rich, policy='fixed', rng=np.random.default_rng(0))
    >>> r['stockout_rate'] == 0.0
    True
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pull parameters — uses fallback if fitted values are None
    mu: float = get_param(config, "lognormal_mu")
    sigma: float = get_param(config, "lognormal_sigma")
    weekday_lambda: np.ndarray = np.array(
        get_param(config, "poisson_lambda_weekday"), dtype=float
    )
    weekend_lambda: np.ndarray = np.array(
        get_param(config, "poisson_lambda_weekend"), dtype=float
    )
    dow_multipliers: np.ndarray = np.array(
        get_param(config, "dow_multipliers"), dtype=float
    )

    cash: float = float(config["initial_cash"])
    refill_amount: float = float(config["refill_amount"])
    fixed_refill_days: int = config["fixed_refill_days"]
    demand_threshold: float = float(config["demand_threshold"])
    weibull_shape: float = config["weibull_shape"]
    weibull_scale: float = config["weibull_scale"]
    sim_days: int = config["sim_days"]

    # Tracking
    stockout_days: int = 0
    stockout_by_dow: list[int] = [0] * 7
    stockout_by_hour: list[int] = [0] * 24
    dispatch_count: int = 0
    cash_history: list[float] = []
    days_since_dispatch: int = 0
    pending_refill_in: int | None = None

    for day in range(sim_days):
        dow: int = day % 7  # 0=Mon, 6=Sun
        is_weekend: bool = dow >= 5
        hourly_lambda = weekend_lambda if is_weekend else weekday_lambda
        multiplier: float = float(dow_multipliers[dow])

        # Truck arrives today
        if pending_refill_in is not None:
            pending_refill_in -= 1
            if pending_refill_in <= 0:
                cash += refill_amount
                pending_refill_in = None
                days_since_dispatch = 0

        # Pre-draw random samples for this day
        scaled_lambdas: np.ndarray = hourly_lambda * multiplier
        poisson_samples: np.ndarray = np.array(
            [rng.poisson(max(lam, 0.0)) for lam in scaled_lambdas],
            dtype=np.int64,
        )
        max_arrivals: int = int(poisson_samples.max()) if poisson_samples.max() > 0 else 1
        lognormal_samples: list[np.ndarray] = [
            rng.lognormal(mu, sigma, size=max(int(poisson_samples[h]), 1))
            for h in range(24)
        ]

        # JIT-compiled inner loop
        cash, day_had_stockout, stockout_hour = _deplete_cash_hourly(
            cash, hourly_lambda, multiplier, mu, sigma,
            poisson_samples, lognormal_samples,
        )

        if day_had_stockout:
            stockout_days += 1
            stockout_by_dow[dow] += 1
            if stockout_hour >= 0:
                stockout_by_hour[stockout_hour] += 1

        cash_history.append(round(cash, 2))
        days_since_dispatch += 1

        # Dispatch decision
        if pending_refill_in is None:
            dispatch: bool = False
            if policy == "fixed" and days_since_dispatch >= fixed_refill_days:
                dispatch = True
            elif policy == "demand" and cash < demand_threshold:
                dispatch = True

            if dispatch:
                delay: float = float(
                    rng.weibull(weibull_shape) * weibull_scale
                )
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
    config: dict[str, Any],
    policy: str = "fixed",
    num_runs: int | None = None,
    base_seed: int = 0,
    label: str = "",
) -> list[dict[str, Any]]:
    """Run simulate_one_year num_runs times and return all results.

    Parameters
    ----------
    config : dict[str, Any]
        Simulation configuration.
    policy : str
        'fixed' or 'demand'.
    num_runs : int or None
        Overrides config['num_runs'] if given.
    base_seed : int
        Run i uses seed base_seed + i for full reproducibility.
    label : str
        Progress label printed to terminal.

    Returns
    -------
    list[dict[str, Any]]
        One result dict per run.

    Examples
    --------
    >>> from src.config import CONFIG
    >>> cfg = {**CONFIG, 'sim_days': 7, 'num_runs': 3}
    >>> results = run_many(cfg, policy='fixed', num_runs=3, label='test')
    >>> len(results) == 3
    True
    >>> all('stockout_rate' in r for r in results)
    True
    >>> all('dispatch_count' in r for r in results)
    True
    """
    n: int = num_runs if num_runs is not None else config["num_runs"]
    results: list[dict[str, Any]] = []
    print_every: int = max(1, n // 5)

    for i in range(n):
        if i % print_every == 0:
            pct: int = int(i / n * 100)
            print(f"  {label}: {pct}% ({i}/{n})")
        rng: np.random.Generator = np.random.default_rng(base_seed + i)
        results.append(simulate_one_year(config, policy=policy, rng=rng))

    print(f"  {label}: 100% ({n}/{n}) done")
    return results
