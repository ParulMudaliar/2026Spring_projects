"""
fit_distributions.py
Phase 1 — Fit all random variable distributions from the Danish ATM dataset.

Run this file directly:
    python -m src.fit_distributions

Then paste the printed values into CONFIG in src/config.py.

Data sources:
    Danish ATM Transactions: kaggle.com/datasets/sparnord/danish-atm-transactions
    ECB Payment Statistics 2022: EU-27 conservative average ATM withdrawal ≈ €150
    (ECB reference rate 7.46 DKK/EUR → DKK 1,100 per transaction)

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_PATH = Path("data/danish_atm.csv")

# Only load the columns we actually need — keeps memory low on a 250MB file
USECOLS = ["hour", "weekday", "atm_id", "atm_status", "service"]


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
    ...     'hour': [10, 14], 'weekday': [1, 6],
    ...     'atm_id': ['A1', 'A1'], 'atm_status': ['Active', 'Active'],
    ...     'service': ['Withdrawal', 'Withdrawal']
    ... })
    >>> df = df[df['atm_status'] == 'Active']
    >>> len(df) == 2
    True
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, usecols=USECOLS, low_memory=False)
    print(f"  Total rows loaded: {len(df):,}")

    # Keep only active ATMs and withdrawal transactions
    df = df[df["atm_status"].str.strip().str.lower() == "active"]
    df = df[df["service"].str.strip().str.lower().str.contains("withdrawal", na=False)]
    print(f"  Rows after filtering (active + withdrawal): {len(df):,}")

    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"])
    return df


def fit_lognormal(eu_mean_dkk: float = 1100.0,
                  sigma: float = 0.82) -> tuple[float, float]:
    """Compute lognormal mu anchored to European average ATM withdrawal in DKK.

    The Danish dataset amount column is synthetic, so we use only the
    sigma shape from literature and anchor the mean to the ECB 2022
    EU-wide average, converted to DKK.

    Parameters
    ----------
    eu_mean_dkk : float
        Conservative average ATM withdrawal in DKK.
        Source: ECB Payment Statistics 2022 — EU-27 total withdrawal value
        ~€1.16 trillion over ~5.46 billion transactions ≈ €212/transaction.
        Conservative estimate used: €150 × 7.46 DKK/EUR ≈ DKK 1,100.
    sigma : float
        Shape parameter from ATM withdrawal literature (Ekinci et al. 2015).

    Returns
    -------
    tuple[float, float]
        (mu, sigma) for np.random.lognormal(mu, sigma).

    Examples
    --------
    >>> mu, s = fit_lognormal(1100.0, 0.82)
    >>> abs(round(mu, 2) - 6.67) < 0.05
    True
    """
    mu = float(np.log(eu_mean_dkk) - (sigma ** 2) / 2)
    print(f"\n--- Lognormal (withdrawal amounts, currency: DKK) ---")
    print(f"  Source: ECB Payment Statistics 2022 (EU-27), conservative estimate")
    print(f"  EU average withdrawal: €150 ≈ DKK {eu_mean_dkk:.0f} (at 7.46 DKK/EUR)")
    print(f"  mu = {mu:.4f},  sigma = {sigma:.4f}")
    print(f"  Implied mean: DKK {np.exp(mu + sigma**2/2):.2f}")
    return mu, sigma


def fit_poisson(df: pd.DataFrame) -> dict[str, list[float]]:
    """Fit hourly Poisson arrival rates from real transaction counts.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe with hour, atm_id, weekday columns.

    Returns
    -------
    dict
        'weekday' and 'weekend' keys, each a list of 24 floats.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'hour': [10,10,14], 'is_weekend': [False,False,True],
    ...                    'atm_id': ['A','A','A'], 'weekday': [1,1,6]})
    >>> result = fit_poisson(df)
    >>> len(result['weekday']) == 24
    True
    """
    print(f"\n--- Poisson (arrival rates) ---")
    print(f"  Source: Danish ATM dataset, hour column")

    def avg_per_hour(subset: pd.DataFrame) -> list[float]:
        counts = (
            subset.groupby(["atm_id", "weekday", "hour"])
            .size()
            .reset_index(name="count")
        )
        avg = counts.groupby("hour")["count"].mean()
        return avg.reindex(range(24), fill_value=0.1).round(4).tolist()

    weekday_lambda = avg_per_hour(df[~df["is_weekend"]])
    weekend_lambda = avg_per_hour(df[df["is_weekend"]])

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
        Monday = 1.0 baseline.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'weekday': ['Monday', 'Monday', 'Saturday'],
    ...                    'atm_id': ['A', 'A', 'A']})
    >>> mults = fit_dow_multipliers(df)
    >>> len(mults) == 7
    True
    """
    print(f"\n--- Day-of-week multipliers ---")
    print(f"  Source: Danish ATM dataset, weekday column")

    ORDER = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]

    daily = (
        df.groupby(["atm_id", "weekday"])
        .size()
        .reset_index(name="count")
    )
    avg = daily.groupby("weekday")["count"].mean()
    avg = avg.reindex(ORDER, fill_value=avg.mean())
    baseline = avg["Monday"]
    multipliers = (avg / baseline).round(4).tolist()

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d, m in zip(days, multipliers):
        bar = "█" * int(m * 10)
        print(f"  {d}: {m:.3f}  {bar}")
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