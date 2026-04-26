TOTAL_HOURS = 24
UNIFORM_SHARE = 8 / 24  # 33.3% — expected if 8 peak hours were uniform


def run_h3(config: dict = CONFIG) -> dict:
    print("\n" + "=" * 55)
    print("PHASE 3 — H3: Peak hour stockout disproportionality")
    print("=" * 55)
    print("Prediction: peak demand hours will account for more than")
    print(f"33.3% of stockouts despite being 33.3% of the day.")

    # Figure out which 8 hours are the busiest based on fitted demand
    from src.config import get_param
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
    print(f"\n  Stockouts by hour of day:")
    for h in range(24):
        bar = "█" * int(pct_by_hour[h] / 2)
        tag = " <- PEAK" if h in peak_hours else ""
        print(f"    {h:02d}:00  {pct_by_hour[h]:5.1f}%  {bar}{tag}")

    print(f"\n  Peak hour stockout share:  {peak_fraction:.2%}")
    print(f"  Expected (uniform):        {UNIFORM_SHARE:.2%}")
    print(f"  Disproportionality ratio:  {ratio:.2f}x")
    print(f"\n  Finding: {finding}")