"""
Phase 3 — Hypothesis 2: A demand-triggered refill policy reduces
stockout rate compared to a fixed 3-day schedule, but increases
dispatch frequency and variability.

Data grounding: Real hourly and daily demand patterns from Danish ATM
dataset calibrate the simulation. Fed Reserve 2022 anchors mean
withdrawal at $198.13.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

import numpy as np
from src.config import CONFIG
from src.simulation import run_many


def run_h2(config: dict = CONFIG) -> dict:
    """Run H2 experiment: fixed schedule vs demand-triggered policy.

    Runs num_runs simulations under each policy and compares:
      - Mean annual stockout rate
      - Mean annual dispatch count
      - Variability (std) of both metrics

    Parameters
    ----------
    config : dict
        Simulation configuration from config.py.

    Returns
    -------
    dict with keys:
        fixed               : dict of stats for fixed policy
        demand              : dict of stats for demand policy
        stockout_reduction  : float  (percentage reduction)
        dispatch_change     : float  (percentage change)
        supported           : bool
        finding             : str

    Examples
    --------
    >>> from src.config import CONFIG
    >>> result = run_h2(CONFIG)
    >>> 'fixed' in result and 'demand' in result
    True
    >>> 0.0 <= result['fixed']['mean_stockout_rate'] <= 1.0
    True
    >>> 0.0 <= result['demand']['mean_stockout_rate'] <= 1.0
    True
    >>> isinstance(result['supported'], bool)
    True
    >>> isinstance(result['finding'], str)
    True
    """
    print("\n" + "=" * 55)
    print("PHASE 3 — H2: Fixed schedule vs demand-triggered policy")
    print("=" * 55)
    print("Prediction: demand-triggered policy reduces stockout rate")
    print("but increases dispatch frequency.")

    print("\n  Running fixed schedule...")
    fixed_results = run_many(
        config,
        policy="fixed",
        label="H2-fixed",
        base_seed=20_000,
    )

    print("\n  Running demand-triggered policy...")
    demand_results = run_many(
        config,
        policy="demand",
        label="H2-demand",
        base_seed=30_000,
    )

    fixed_stats = _summarize(fixed_results)
    demand_stats = _summarize(demand_results)

    fixed_rate = fixed_stats["mean_stockout_rate"]
    demand_rate = demand_stats["mean_stockout_rate"]

    stockout_reduction = (
        (fixed_rate - demand_rate) / fixed_rate * 100
        if fixed_rate > 0 else 0.0
    )

    fixed_dispatch = fixed_stats["mean_dispatch_count"]
    demand_dispatch = demand_stats["mean_dispatch_count"]

    dispatch_change = (
        (demand_dispatch - fixed_dispatch) / fixed_dispatch * 100
        if fixed_dispatch > 0 else 0.0
    )

    supported = stockout_reduction > 5.0

    finding = (
        f"Demand-triggered policy changed stockout rate by "
        f"{stockout_reduction:.1f}% "
        f"({fixed_rate:.2%} -> {demand_rate:.2%}) "
        f"and dispatch count by {dispatch_change:+.1f}% "
        f"({fixed_dispatch:.0f} -> {demand_dispatch:.0f} trips/year). "
        f"H2 {'SUPPORTED' if supported else 'NOT SUPPORTED'}."
    )

    _print_results(fixed_stats, demand_stats,
                   stockout_reduction, dispatch_change, finding)

    return {
        "fixed": fixed_stats,
        "demand": demand_stats,
        "stockout_reduction": round(stockout_reduction, 4),
        "dispatch_change": round(dispatch_change, 4),
        "supported": supported,
        "finding": finding,
    }


def _summarize(results: list[dict]) -> dict:
    """Compute summary statistics from a list of simulation results.

    Parameters
    ----------
    results : list[dict]
        Output from run_many().

    Returns
    -------
    dict with keys:
        mean_stockout_rate  : float
        std_stockout_rate   : float
        p5_stockout_rate    : float
        p95_stockout_rate   : float
        mean_dispatch_count : float
        std_dispatch_count  : float
        all_rates           : list[float]

    Examples
    --------
    >>> results = [{'stockout_rate': 0.1, 'dispatch_count': 100,
    ...             'stockout_by_dow': [0]*7,
    ...             'stockout_by_hour': [0]*24,
    ...             'stockout_days': 10,
    ...             'cash_history': []}]
    >>> s = _summarize(results)
    >>> s['mean_stockout_rate'] == 0.1
    True
    """
    rates = [r["stockout_rate"] for r in results]
    dispatches = [r["dispatch_count"] for r in results]

    return {
        "mean_stockout_rate": round(float(np.mean(rates)), 6),
        "std_stockout_rate": round(float(np.std(rates)), 6),
        "p5_stockout_rate": round(float(np.percentile(rates, 5)), 6),
        "p95_stockout_rate": round(float(np.percentile(rates, 95)), 6),
        "mean_dispatch_count": round(float(np.mean(dispatches)), 2),
        "std_dispatch_count": round(float(np.std(dispatches)), 2),
        "all_rates": rates,
    }


def _print_results(
    fixed: dict,
    demand: dict,
    stockout_reduction: float,
    dispatch_change: float,
    finding: str,
) -> None:
    """Print formatted H2 comparison table to terminal.

    Parameters
    ----------
    fixed : dict
        Summary stats for fixed policy.
    demand : dict
        Summary stats for demand policy.
    stockout_reduction : float
        Percentage reduction in stockout rate.
    dispatch_change : float
        Percentage change in dispatch count.
    finding : str
        Plain-English result statement.

    Examples
    --------
    >>> fixed = {'mean_stockout_rate': 0.10, 'std_stockout_rate': 0.02,
    ...          'p5_stockout_rate': 0.06, 'p95_stockout_rate': 0.15,
    ...          'mean_dispatch_count': 120.0, 'std_dispatch_count': 3.0,
    ...          'all_rates': []}
    >>> demand = {'mean_stockout_rate': 0.07, 'std_stockout_rate': 0.02,
    ...           'p5_stockout_rate': 0.04, 'p95_stockout_rate': 0.11,
    ...           'mean_dispatch_count': 135.0, 'std_dispatch_count': 5.0,
    ...           'all_rates': []}
    >>> _print_results(fixed, demand, 30.0, 12.5, 'H2 SUPPORTED.')
    """
    print(f"\n  {'Policy':<20} {'Mean stockout':>14} "
          f"{'Dispatches/yr':>14}")
    print("  " + "-" * 50)
    print(f"  {'Fixed 3-day':<20} "
          f"{fixed['mean_stockout_rate']:>13.2%} "
          f"{fixed['mean_dispatch_count']:>14.1f}")
    print(f"  {'Demand-triggered':<20} "
          f"{demand['mean_stockout_rate']:>13.2%} "
          f"{demand['mean_dispatch_count']:>14.1f}")
    print(f"\n  Stockout rate change:  {stockout_reduction:+.1f}%")
    print(f"  Dispatch count change: {dispatch_change:+.1f}%")
    print(f"\n  Finding: {finding}")


if __name__ == "__main__":
    run_h2()