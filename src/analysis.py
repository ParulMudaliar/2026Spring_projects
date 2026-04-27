"""
Generates plots and saves results for all phases.

Saves all plots to outputs/ folder.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_control_run(phase2: dict) -> None:
    """Plot histogram of stockout rates from Phase 2 control run.

    Parameters
    ----------
    phase2 : dict
        Output from phase2_control.run_control().

    Examples
    --------
    >>> plot_control_run({'all_rates': [0.1, 0.12, 0.09],
    ...                   'mean_stockout_rate': 0.103})
    """
    rates = phase2["all_rates"]
    mean = phase2["mean_stockout_rate"]

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

def plot_h1(h1: dict) -> None:
    """Plot stockout distribution by day of week for H1.

    Parameters
    ----------
    h1 : dict
        Output from h1.run_h1().

    Examples
    --------
    >>> plot_h1({'stockouts_by_dow_pct': [14.0]*7,
    ...          'weekend_stockout_fraction': 0.286,
    ...          'expected_weekend_fraction': 0.286,
    ...          'disproportionality_ratio': 1.0})
    """
    pcts = h1["stockouts_by_dow_pct"]
    colors = ["#B5D4F4"] * 5 + ["#534AB7", "#534AB7"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(DAYS, pcts, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(100 / 7, color="#E8593C", linestyle="--",
               linewidth=1.5,
               label=f"Uniform baseline ({100/7:.1f}%)")

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8
        )

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


def plot_h2(h2: dict) -> None:
    """Plot side-by-side stockout rate distributions for H2.

    Parameters
    ----------
    h2 : dict
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
        rates = h2[key]["all_rates"]
        mean = h2[key]["mean_stockout_rate"]
        ax.hist(rates, bins=50, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(mean, color="#E8593C", linewidth=2,
                   label=f"Mean = {mean:.2%}")
        ax.set_xlabel("Annual stockout rate")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )

    axes[0].set_ylabel("Number of runs")
    fig.suptitle(
        f"H2 — Policy comparison\n"
        f"Stockout change: {h2['stockout_reduction']:+.1f}%  |  "
        f"Dispatch change: {h2['dispatch_change']:+.1f}%",
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(OUTPUTS / "h2_policy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: outputs/h2_policy_comparison.png")
