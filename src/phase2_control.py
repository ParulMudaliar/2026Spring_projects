def run_control(config: dict = CONFIG) -> dict:
    print("\n" + "=" * 55)
    print("PHASE 2 — Control run (fixed 3-day schedule)")
    print("=" * 55)

    results = run_many(
        config,
        policy="fixed",
        label="Control",
        base_seed=0,
    )

    rates = [r["stockout_rate"] for r in results]
    dispatches = [r["dispatch_count"] for r in results]

    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates))
    min_rate = float(np.min(rates))
    max_rate = float(np.max(rates))
    mean_dispatches = float(np.mean(dispatches))

    benchmark_pass = (
        config["benchmark_min"] <= mean_rate <= config["benchmark_max"]
    )

    if benchmark_pass:
        verdict = (
            f"PASS — mean stockout rate {mean_rate:.2%} is within "
            f"benchmark range "
            f"({config['benchmark_min']:.0%}–{config['benchmark_max']:.0%}). "
            f"Proceeding to Phase 3."
        )
    else:
        verdict = (
            f"WARNING — mean stockout rate {mean_rate:.2%} is outside "
            f"benchmark range "
            f"({config['benchmark_min']:.0%}–{config['benchmark_max']:.0%}). "
            f"Re-examine distribution parameters before Phase 3."
        )

    _print_results(mean_rate, std_rate, min_rate, max_rate,
                   mean_dispatches, verdict)

    return {
        "mean_stockout_rate": mean_rate,
        "std_stockout_rate": std_rate,
        "min_stockout_rate": min_rate,
        "max_stockout_rate": max_rate,
        "mean_dispatch_count": mean_dispatches,
        "benchmark_pass": benchmark_pass,
        "verdict": verdict,
        "all_rates": rates,
    }


def _print_results(mean_rate, std_rate, min_rate, max_rate, mean_dispatches, verdict):
    print(f"\n  Mean stockout rate:   {mean_rate:.2%}")
    print(f"  Std stockout rate:    {std_rate:.4f}")
    print(f"  Range:                {min_rate:.2%} — {max_rate:.2%}")
    print(f"  Mean dispatches/year: {mean_dispatches:.1f}")
    print(f"\n  {verdict}")