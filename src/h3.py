import numpy as np
from src.config import CONFIG, get_param
from src.simulation import run_many
"""
h3.py
Phase 3 — Hypothesis 3.

ATM transactions in the Danish dataset cluster heavily in certain hours,
so we'd expect stockouts to follow the same pattern rather than spread
evenly across the day. This file tests whether peak hours — the top 8
by average demand — account for more than their fair share (33.3%) of
stockout events.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
AI Disclosure: Claude (Anthropic) assisted in drafting this code.
"""

TOTAL_HOURS = 24
UNIFORM_SHARE = 8 / 24  # 33.3% — expected if 8 peak hours were uniform


def run_h3(config: dict = CONFIG) -> dict:
    """Test whether stockouts cluster in high-demand hours.

    Picks the 8 busiest hours from the fitted Poisson lambdas, runs
    num_runs simulations under the fixed policy, and checks whether
    those hours attract more than their uniform share (8/24 = 33.3%)
    of stockout events.

    Parameters
    ----------
    config : dict
        Simulation config from config.py.

    Returns
    -------
    dict with keys:
        peak_hours               : list[int]
        peak_stockout_fraction   : float
        expected_peak_fraction   : float  (always 8/24)
        disproportionality_ratio : float
        stockouts_by_hour        : list[int]
        stockouts_by_hour_pct    : list[float]
        total_stockouts          : int
        supported                : bool
        finding                  : str

    Examples
    --------
    >>> result = run_h3(CONFIG)
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
    print("Prediction: peak demand hours will account for more than")
    print(f"33.3% of stockouts despite being 33.3% of the day.")

    # Figure out which 8 hours are the busiest based on fitted demand
    weekday_lambda = np.array(
        get_param(config, "poisson_lambda_weekday"), dtype=float
    )
    weekend_lambda = np.array(
        get_param(config, "poisson_lambda_weekend"), dtype=float
    )
    avg_lambda = (weekday_lambda + weekend_lambda) / 2
    peak_hours = sorted(
        np.argsort(avg_lambda)[-8:].tolist()
    )

    print(f"\n  Peak hours identified from data: {peak_hours}")

    results = run_many(
        config,
        policy="fixed",
        label="H3",
        base_seed=40_000,
    )

    # Sum up per-hour stockouts across all simulation runs
    total_by_hour = [0] * 24
    for r in results:
        for i in range(24):
            total_by_hour[i] += r["stockout_by_hour"][i]

    total_stockouts = sum(total_by_hour)

    if total_stockouts == 0:
        return _zero_stockout_result(peak_hours)

    peak_stockouts = sum(total_by_hour[h] for h in peak_hours)
    peak_fraction = peak_stockouts / total_stockouts
    ratio = peak_fraction / UNIFORM_SHARE

    pct_by_hour = [
        round(total_by_hour[i] / total_stockouts * 100, 2)
        for i in range(24)
    ]

    supported = ratio > 1.2

    finding = (
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

def _zero_stockout_result(peak_hours: list[int]) -> dict:
    """Fallback for when no stockouts occurred across all runs.

    Shouldn't happen under normal parameters, but better to return
    a clean dict than crash downstream.

    Parameters
    ----------
    peak_hours : list[int]
        Identified peak hours from lambda fitting.

    Returns
    -------
    dict
        Result dict with zero values and unsupported finding.

    Examples
    --------
    >>> result = _zero_stockout_result([10, 11, 12])
    >>> result['total_stockouts'] == 0
    True
    >>> result['supported'] == False
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
    """
    print(f"\n  Stockouts by hour of day:")
    for h in range(24):
        bar = "█" * int(pct_by_hour[h] / 2)
        tag = " <- PEAK" if h in peak_hours else ""
        print(f"    {h:02d}:00  {pct_by_hour[h]:5.1f}%  {bar}{tag}")

    print(f"\n  Peak hour stockout share:  {peak_fraction:.2%}")
    print(f"  Expected (uniform):        {UNIFORM_SHARE:.2%}")
    print(f"  Disproportionality ratio:  {ratio:.2f}x")
    print(f"\n  Finding: {finding}")