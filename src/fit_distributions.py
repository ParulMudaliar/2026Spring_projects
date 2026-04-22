"""
fit_distributions.py
Phase 1 — Fit all random variable distributions from the Danish ATM dataset.

Run this file directly:
    python -m src.fit_distributions

Data sources:
    Danish ATM Transactions: kaggle.com/datasets/sparnord/danish-atm-transactions
    Federal Reserve Payments Study 2022: average withdrawal = $198.13

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
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
