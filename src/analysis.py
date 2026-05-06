"""
Generates plots and saves results for all phases.

Saves all plots to outputs/ folder.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/testing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

OUTPUTS: Path = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

DAYS: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_control_run(phase2: dict[str, Any]) -> None:
    """Plot histogram of stockout rates from Phase 2 control run.

    Parameters
    ----------
    phase2 : dict[str, Any]
        Output from phase2_control.run_control().

    Examples
    --------
    >>> plot_control_run({'all_rates': [0.1, 0.12, 0.09, 0.11, 0.10],
    ...                   'mean_stockout_rate': 0.104})
    """
    rates: list[float] = phase2["all_rates"]
    mean: float = phase2["mean_stockout_rate"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rates, bins=50, color="#534AB7", alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(mean, color="#E8593C", linewidth=2,
               label=f"Mean = {mean:.2%}")
    ax.axvline(0.05, color="gray", linestyle="--",
               linewidth=1, label="Benchmark lower (5%)")
    ax.axvline(0.20, color="gray", linestyle=":",
               linewidth=1, label="Benchmark upper (20%)")
    ax.set_xlabel("Annual stockout rate")
    ax.set_ylabel("Number of runs")
    ax.set_title("Phase 2 — Control run: stockout rate distribution\n"
                 "(fixed 3-day schedule, 10,000 runs)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.tight_layout()
    fig.savefig(OUTPUTS / "phase2_control.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: outputs/phase2_control.png")


def plot_h1(h1: dict[str, Any]) -> None:
    """Plot stockout distribution by day of week for H1.

    Parameters
    ----------
    h1 : dict[str, Any]
        Output from h1.run_h1().

    Examples
    --------
    >>> plot_h1({'stockouts_by_dow_pct': [14.0]*7,
    ...          'weekend_stockout_fraction': 0.286,
    ...          'expected_weekend_fraction': 0.286,
    ...          'disproportionality_ratio': 1.0})
    """
    pcts: list[float] = h1["stockouts_by_dow_pct"]
    colors: list[str] = ["#B5D4F4"] * 5 + ["#534AB7", "#534AB7"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(DAYS, pcts, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(100 / 7, color="#E8593C", linestyle="--",
               linewidth=1.5,
               label=f"Uniform baseline ({100/7:.1f}%)")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("% of total stockouts")
    ax.set_title(
        f"H1 — Stockout distribution by day of week\n"
        f"Weekend share: {h1['weekend_stockout_fraction']:.1%} "
        f"(expected {h1['expected_weekend_fraction']:.1%}, "
        f"ratio: {h1['disproportionality_ratio']:.2f}x)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUTS / "h1_dow_stockouts.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: outputs/h1_dow_stockouts.png")


def plot_h2(h2: dict[str, Any]) -> None:
    """Plot side-by-side stockout rate distributions for H2.

    Parameters
    ----------
    h2 : dict[str, Any]
        Output from h2.run_h2().

    Examples
    --------
    >>> plot_h2({'fixed': {'all_rates': [0.1, 0.12],
    ...                    'mean_stockout_rate': 0.11},
    ...          'demand': {'all_rates': [0.07, 0.08],
    ...                     'mean_stockout_rate': 0.075},
    ...          'stockout_reduction': 31.8,
    ...          'dispatch_change': 12.5})
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, key, label, color in [
        (axes[0], "fixed",  "Fixed 3-day schedule",   "#B5D4F4"),
        (axes[1], "demand", "Demand-triggered policy", "#534AB7"),
    ]:
        rates: list[float] = h2[key]["all_rates"]
        mean: float = h2[key]["mean_stockout_rate"]
        ax.hist(rates, bins=50, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(mean, color="#E8593C", linewidth=2,
                   label=f"Mean = {mean:.2%}")
        ax.set_xlabel("Annual stockout rate")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[0].set_ylabel("Number of runs")
    fig.suptitle(
        f"H2 — Policy comparison\n"
        f"Stockout change: {h2['stockout_reduction']:+.1f}%  |  "
        f"Dispatch change: {h2['dispatch_change']:+.1f}%", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS / "h2_policy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: outputs/h2_policy_comparison.png")


def plot_h3(h3: dict[str, Any]) -> None:
    """Plot stockout distribution by hour of day for H3.

    Parameters
    ----------
    h3 : dict[str, Any]
        Output from h3.run_h3().

    Examples
    --------
    >>> plot_h3({'stockouts_by_hour_pct': [4.0]*24,
    ...          'peak_hours': [10,11,12,13,14,15,16,17],
    ...          'peak_stockout_fraction': 0.4,
    ...          'expected_peak_fraction': 0.333,
    ...          'disproportionality_ratio': 1.2})
    """
    pcts: list[float] = h3["stockouts_by_hour_pct"]
    peak_hours: list[int] = h3["peak_hours"]
    colors: list[str] = [
        "#534AB7" if h in peak_hours else "#B5D4F4"
        for h in range(24)
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(24), pcts, color=colors,
           edgecolor="white", linewidth=0.3)
    ax.axhline(100 / 24, color="#E8593C", linestyle="--",
               linewidth=1.5,
               label=f"Uniform baseline ({100/24:.1f}%)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of total stockouts")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_title(
        f"H3 — Stockout distribution by hour of day\n"
        f"Peak hour share: {h3['peak_stockout_fraction']:.1%} "
        f"(expected {h3['expected_peak_fraction']:.1%}, "
        f"ratio: {h3['disproportionality_ratio']:.2f}x)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUTS / "h3_hour_stockouts.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: outputs/h3_hour_stockouts.png")


def save_results(
    phase2: dict[str, Any],
    h1: dict[str, Any],
    h2: dict[str, Any],
    h3: dict[str, Any],
) -> None:
    """Save all results to a JSON file in outputs/.

    Parameters
    ----------
    phase2 : dict[str, Any]
        Phase 2 control run results.
    h1 : dict[str, Any]
        H1 results.
    h2 : dict[str, Any]
        H2 results.
    h3 : dict[str, Any]
        H3 results.

    Examples
    --------
    >>> save_results({'mean_stockout_rate': 0.1, 'all_rates': []},
    ...     {'finding': 'H1', 'supported': True,
    ...      'weekend_stockout_fraction': 0.4,
    ...      'weekday_stockout_fraction': 0.6,
    ...      'expected_weekend_fraction': 0.286,
    ...      'disproportionality_ratio': 1.4,
    ...      'stockouts_by_dow': [0]*7,
    ...      'stockouts_by_dow_pct': [0.0]*7,
    ...      'total_stockouts': 0},
    ...     {'finding': 'H2', 'supported': True,
    ...      'stockout_reduction': 30.0, 'dispatch_change': 12.0,
    ...      'fixed': {'mean_stockout_rate': 0.1, 'std_stockout_rate': 0.02,
    ...                'p5_stockout_rate': 0.06, 'p95_stockout_rate': 0.15,
    ...                'mean_dispatch_count': 120.0, 'std_dispatch_count': 3.0,
    ...                'all_rates': []},
    ...      'demand': {'mean_stockout_rate': 0.07, 'std_stockout_rate': 0.02,
    ...                 'p5_stockout_rate': 0.04, 'p95_stockout_rate': 0.11,
    ...                 'mean_dispatch_count': 135.0, 'std_dispatch_count': 5.0,
    ...                 'all_rates': []}},
    ...     {'finding': 'H3', 'supported': True,
    ...      'peak_hours': [10,11], 'peak_stockout_fraction': 0.5,
    ...      'expected_peak_fraction': 0.333, 'disproportionality_ratio': 1.5,
    ...      'stockouts_by_hour': [0]*24, 'stockouts_by_hour_pct': [0.0]*24,
    ...      'total_stockouts': 0})
    """
    p2: dict[str, Any] = {k: v for k, v in phase2.items() if k != "all_rates"}
    h2_clean: dict[str, Any] = {
        k: v for k, v in h2.items() if k not in ("fixed", "demand")
    }
    h2_clean["fixed"] = {
        k: v for k, v in h2["fixed"].items() if k != "all_rates"
    }
    h2_clean["demand"] = {
        k: v for k, v in h2["demand"].items() if k != "all_rates"
    }
    all_results: dict[str, Any] = {
        "phase2_control": p2,
        "h1_weekend_stockouts": h1,
        "h2_policy_comparison": h2_clean,
        "h3_peak_hour_stockouts": h3,
    }
    out: Path = OUTPUTS / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: outputs/results.json")


def print_summary(
    phase2: dict[str, Any],
    h1: dict[str, Any],
    h2: dict[str, Any],
    h3: dict[str, Any],
) -> None:
    """Print final summary of all findings to terminal.

    Parameters
    ----------
    phase2 : dict[str, Any]
        Phase 2 control run results.
    h1 : dict[str, Any]
        H1 results.
    h2 : dict[str, Any]
        H2 results.
    h3 : dict[str, Any]
        H3 results.

    Examples
    --------
    >>> print_summary(
    ...     {'verdict': 'PASS', 'mean_stockout_rate': 0.1},
    ...     {'finding': 'H1 SUPPORTED'},
    ...     {'finding': 'H2 SUPPORTED'},
    ...     {'finding': 'H3 SUPPORTED'})
    <BLANKLINE>
    =======================================================
    FINAL SUMMARY
    =======================================================
    <BLANKLINE>
    Phase 2: PASS
    <BLANKLINE>
    H1: H1 SUPPORTED
    <BLANKLINE>
    H2: H2 SUPPORTED
    <BLANKLINE>
    H3: H3 SUPPORTED
    <BLANKLINE>
    Plots saved to outputs/
    Results saved to outputs/results.json
    """
    print("\n" + "=" * 55)
    print("FINAL SUMMARY")
    print("=" * 55)
    print(f"\nPhase 2: {phase2['verdict']}")
    print(f"\nH1: {h1['finding']}")
    print(f"\nH2: {h2['finding']}")
    print(f"\nH3: {h3['finding']}")
    print("\nPlots saved to outputs/")
    print("Results saved to outputs/results.json")