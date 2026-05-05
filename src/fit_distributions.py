"""
fit_distributions.py
Phase 1 — Fit all random variable distributions from the Danish ATM dataset.

Run this file directly:
    python -m src.fit_distributions

Data sources:
    Danish ATM Transactions: kaggle.com/datasets/sparnord/danish-atm-transactions
    Federal Reserve Payments Study 2022: average withdrawal = $198.13

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

DATA_PATH: Path = Path("data/danish_atm.csv")
USECOLS: list[str] = ["hour", "weekday", "atm_id", "atm_status", "service"]


def load_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    """Load only required columns from the Danish ATM dataset.

    Parameters
    ----------
    filepath : Path
        Path to danish_atm.csv.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only withdrawal transactions.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'hour': [10, 14], 'weekday': ['Monday', 'Saturday'],
    ...     'atm_id': ['A1', 'A1'], 'atm_status': ['Active', 'Active'],
    ...     'service': ['Withdrawal', 'Withdrawal']
    ... })
    >>> df = df[df['atm_status'].str.strip().str.lower() == 'active']
    >>> df = df[df['service'].str.strip().str.lower().str.contains('withdrawal', na=False)]
    >>> len(df) == 2
    True
    """
    print(f"Loading data from {filepath}...")
    df: pd.DataFrame = pd.read_csv(filepath, usecols=USECOLS, low_memory=False)
    print(f"  Total rows loaded: {len(df):,}")
    df = df[df["atm_status"].str.strip().str.lower() == "active"]
    df = df[df["service"].str.strip().str.lower().str.contains("withdrawal", na=False)]
    print(f"  Rows after filtering (active + withdrawal): {len(df):,}")
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"])
    return df


def fit_lognormal(
    fed_mean: float = 198.13,
    sigma: float = 0.82,
) -> tuple[float, float]:
    """Compute lognormal mu anchored to Federal Reserve mean withdrawal.

    Parameters
    ----------
    fed_mean : float
        Average ATM withdrawal in USD.
    sigma : float
        Shape parameter from ATM literature (Ekinci et al. 2015).

    Returns
    -------
    tuple[float, float]
        (mu, sigma) for np.random.lognormal(mu, sigma).

    Examples
    --------
    >>> mu, s = fit_lognormal(198.13, 0.82)
    >>> abs(round(mu, 2) - 4.95) < 0.05
    True
    >>> s == 0.82
    True

    >>> mu2, _ = fit_lognormal(100.0, 0.5)
    >>> abs(np.exp(mu2 + 0.5**2 / 2) - 100.0) < 0.01
    True
    """
    mu: float = float(np.log(fed_mean) - (sigma ** 2) / 2)
    print(f"\n--- Lognormal (withdrawal amounts) ---")
    print(f"  mu = {mu:.4f},  sigma = {sigma:.4f}")
    print(f"  Implied mean: ${np.exp(mu + sigma**2/2):.2f}")
    return mu, sigma


def fit_poisson(df: pd.DataFrame) -> dict[str, list[float]]:
    """Fit hourly Poisson arrival rates from real transaction counts.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe with hour, atm_id, weekday columns.

    Returns
    -------
    dict[str, list[float]]
        'weekday' and 'weekend' keys, each a list of 24 floats.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'hour': [10, 10, 14],
    ...     'is_weekend': [False, False, True],
    ...     'atm_id': ['A', 'A', 'A'],
    ...     'weekday': ['Monday', 'Monday', 'Saturday']
    ... })
    >>> result = fit_poisson(df)
    >>> len(result['weekday']) == 24
    True
    >>> len(result['weekend']) == 24
    True
    >>> all(isinstance(v, float) for v in result['weekday'])
    True
    """
    print(f"\n--- Poisson (arrival rates) ---")

    def avg_per_hour(subset: pd.DataFrame) -> list[float]:
        """Return mean transaction count per hour across all ATMs and days in subset."""
        counts = (subset.groupby(["atm_id", "weekday", "hour"])
                  .size().reset_index(name="count"))
        avg = counts.groupby("hour")["count"].mean()
        return avg.reindex(range(24), fill_value=0.1).round(4).tolist()

    weekday_lambda: list[float] = avg_per_hour(df[~df["is_weekend"]])
    weekend_lambda: list[float] = avg_per_hour(df[df["is_weekend"]])
    print(f"  Weekday peak lambda: {max(weekday_lambda):.2f}")
    print(f"  Weekend peak lambda: {max(weekend_lambda):.2f}")
    return {"weekday": weekday_lambda, "weekend": weekend_lambda}


def fit_dow_multipliers(df: pd.DataFrame) -> list[float]:
    """Compute day-of-week demand multipliers relative to Monday.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe with weekday and atm_id columns.

    Returns
    -------
    list[float]
        7 multipliers [Mon, Tue, Wed, Thu, Fri, Sat, Sun].

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'weekday': ['Monday', 'Monday', 'Saturday'],
    ...     'atm_id': ['A', 'A', 'A']
    ... })
    >>> mults = fit_dow_multipliers(df)
    >>> len(mults) == 7
    True
    >>> mults[0] == 1.0
    True
    """
    print(f"\n--- Day-of-week multipliers ---")
    ORDER: list[str] = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
    ]
    daily = (df.groupby(["atm_id", "weekday"]).size()
             .reset_index(name="count"))
    avg = daily.groupby("weekday")["count"].mean()
    avg = avg.reindex(ORDER, fill_value=avg.mean())
    baseline: float = float(avg["Monday"])
    multipliers: list[float] = (avg / baseline).round(4).tolist()
    days: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d, m in zip(days, multipliers):
        print(f"  {d}: {m:.3f}")
    return multipliers


if __name__ == "__main__":
    df = load_data()
    mu, sigma = fit_lognormal()
    poisson = fit_poisson(df)
    dow = fit_dow_multipliers(df)
    print("\n" + "=" * 55)
    print("PASTE THESE INTO src/config.py:")
    print("=" * 55)
    print(f'  "lognormal_mu": {mu},')
    print(f'  "lognormal_sigma": {sigma},')
    print(f'  "poisson_lambda_weekday": {poisson["weekday"]},')
    print(f'  "poisson_lambda_weekend": {poisson["weekend"]},')
    print(f'  "dow_multipliers": {dow},')
