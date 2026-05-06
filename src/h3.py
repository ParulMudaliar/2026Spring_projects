"""
h3.py
Phase 3 — Hypothesis 3: Peak demand hours account for more than
their fair share of stockout events.

ATM transactions in the Danish dataset cluster heavily in certain hours,
so we test whether the top 8 hours by average demand account for more
than 33.3% (8/24) of stockout events.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

import numpy as np
from typing import Any
from src.config import CONFIG, get_param
from src.simulation import run_many

TOTAL_HOURS: int = 24
UNIFORM_SHARE: float = 8 / 24  # 33.3%


def run_h3(config: dict[str, Any] = CONFIG) -> dict[str, Any]:
    """Test whether stockouts cluster in high-demand hours.

    Picks the 8 busiest hours from the fitted Poisson lambdas, runs
    num_runs simulations under the fixed policy, and checks whether
    those hours attract more than their uniform share (8/24 = 33.3%).

    Parameters
    ----------
    config : dict[str, Any]
        Simulation config from config.py.

    Returns
    -------
    dict[str, Any]
        Keys: peak_hours, peak_stockout_fraction, expected_peak_fraction,
        disproportionality_ratio, stockouts_by_hour, stockouts_by_hour_pct,
        total_stockouts, supported, finding.

    Examples
    --------
    >>> from src.config import CONFIG
    >>> cfg = {**CONFIG, 'sim_days': 30, 'num_runs': 5,
    ...        'benchmark_min': 0.0, 'benchmark_max': 1.0}
    >>> result = run_h3(cfg)
    >>> 0.0 <= result['peak_stockout_fraction'] <= 1.0
    True
    >>> result['expected_peak_fraction'] == 8 / 24
    True
    >>> isinstance(result['supported'], bool)
    True
    >>> len(result['stockouts_by_hour']) == 24
    True
    >>> len(result['peak_hours']) == 8
    True
    """
    print("\n" + "=" * 55)
    print("PHASE 3 — H3: Peak hour stockout disproportionality")
    print("=" * 55)

    weekday_lambda: np.ndarray = np.array(
        get_param(config, "poisson_lambda_weekday"), dtype=float
    )
    weekend_lambda: np.ndarray = np.array(
        get_param(config, "poisson_lambda_weekend"), dtype=float
    )
    avg_lambda: np.ndarray = (weekday_lambda + weekend_lambda) / 2
    peak_hours: list[int] = sorted(np.argsort(avg_lambda)[-8:].tolist())

    print(f"\n  Peak hours identified from data: {peak_hours}")

    results: list[dict[str, Any]] = run_many(
        config, policy="fixed", label="H3", base_seed=40_000,
    )

    total_by_hour: list[int] = [0] * 24
    for r in results:
        for i in range(24):
            total_by_hour[i] += r["stockout_by_hour"][i]

    total_stockouts: int = sum(total_by_hour)

    if total_stockouts == 0:
        return _zero_stockout_result(peak_hours)

    peak_stockouts: int = sum(total_by_hour[h] for h in peak_hours)
    peak_fraction: float = peak_stockouts / total_stockouts
    ratio: float = peak_fraction / UNIFORM_SHARE

    pct_by_hour: list[float] = [
        round(total_by_hour[i] / total_stockouts * 100, 2)
        for i in range(24)
    ]

    supported: bool = ratio > 1.2

    finding: str = (
        f"{peak_fraction:.1%} of stockouts occur during peak hours "
        f"{peak_hours} vs expected {UNIFORM_SHARE:.1%} "
        f"(ratio: {ratio:.2f}x). "
        f"H3 {'SUPPORTED' if supported else 'NOT SUPPORTED'}."
    )

    _print_results(total_by_hour, pct_by_hour, peak_hours,
                   peak_fraction, ratio, finding)

    return {
        "peak_hours": peak_hours,
        "peak_stockout_fraction": round(peak_fraction, 6),
        "expected_peak_fraction": UNIFORM_SHARE,
        "disproportionality_ratio": round(ratio, 4),
        "stockouts_by_hour": total_by_hour,
        "stockouts_by_hour_pct": pct_by_hour,
        "total_stockouts": total_stockouts,
        "supported": supported,
        "finding": finding,
    }


def _zero_stockout_result(peak_hours: list[int]) -> dict[str, Any]:
    """Fallback for when no stockouts occurred across all runs.

    Parameters
    ----------
    peak_hours : list[int]
        Identified peak hours from lambda fitting.

    Returns
    -------
    dict[str, Any]
        Result dict with zero values and unsupported finding.

    Examples
    --------
    >>> result = _zero_stockout_result([10, 11, 12, 13, 14, 15, 16, 17])
    >>> result['total_stockouts'] == 0
    True
    >>> result['supported'] == False
    True
    >>> len(result['stockouts_by_hour']) == 24
    True
    >>> len(result['peak_hours']) == 8
    True
    """
    return {
        "peak_hours": peak_hours,
        "peak_stockout_fraction": 0.0,
        "expected_peak_fraction": UNIFORM_SHARE,
        "disproportionality_ratio": 0.0,
        "stockouts_by_hour": [0] * 24,
        "stockouts_by_hour_pct": [0.0] * 24,
        "total_stockouts": 0,
        "supported": False,
        "finding": "No stockouts observed. Check distribution parameters.",
    }


def _print_results(
        total_by_hour: list[int],
        pct_by_hour: list[float],
        peak_hours: list[int],
        peak_fraction: float,
        ratio: float,
        finding: str,
) -> None:
    """Print the H3 breakdown to the terminal.

    Parameters
    ----------
    total_by_hour : list[int]
        Total stockout count per hour of day.
    pct_by_hour : list[float]
        Percentage of total stockouts per hour.
    peak_hours : list[int]
        The 8 identified peak hours.
    peak_fraction : float
        Fraction of stockouts in peak hours.
    ratio : float
        Disproportionality ratio vs uniform expectation.
    finding : str
        Plain-English result statement.

    Examples
    --------
    >>> _print_results([0]*24, [0.0]*24, [10,11,12,13,14,15,16,17],
    ...                0.5, 1.5, 'H3 SUPPORTED.')
    <BLANKLINE>
      Stockouts by hour of day:
        ...
    """
    print(f"\n  Stockouts by hour of day:")
    for h in range(24):
        bar: str = "█" * int(pct_by_hour[h] / 2)
        tag: str = " <- PEAK" if h in peak_hours else ""
        print(f"    {h:02d}:00  {pct_by_hour[h]:5.1f}%  {bar}{tag}")
    print(f"\n  Peak hour stockout share:  {peak_fraction:.2%}")
    print(f"  Expected (uniform):        {UNIFORM_SHARE:.2%}")
    print(f"  Disproportionality ratio:  {ratio:.2f}x")
    print(f"\n  Finding: {finding}")
