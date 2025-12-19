"""
Data loading utilities for the Steam recommendation project.

All functions here load raw CSV files from data/raw and return pandas
DataFrames with minimal or no preprocessing. Higher level logic
(filtering, encoding) lives in other modules.
"""

from pathlib import Path
import pandas as pd

# Project root is two levels above this file:
#   src/data/loader.py -> src/ -> project root
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"


def load_interactions(train: bool = True) -> pd.DataFrame:
    """
    Load user item interaction data.

    Parameters
    ----------
    train : bool, default True
        If True, load train_interactions.csv.
        If False, load test_interactions_in.csv
        (Codabench fold-in interactions for the public test users).

    Returns
    -------
    pd.DataFrame
        Interaction data with at least columns ['user_id', 'item_id'].
        Some variants also include 'playtime'.
    """
    filename = "train_interactions.csv" if train else "test_interactions_in.csv"
    return pd.read_csv(DATA_DIR / filename)


def load_games() -> pd.DataFrame:
    """
    Load core game metadata.

    Returns
    -------
    pd.DataFrame
        games.csv with one row per item_id, including genres,
        publisher, and other metadata fields.
    """
    return pd.read_csv(DATA_DIR / "games.csv")


def load_extended_games() -> pd.DataFrame:
    """
    Load extended game metadata.

    Returns
    -------
    pd.DataFrame
        extended_games.csv with additional information such as platform
        or average playtime.
    """
    return pd.read_csv(DATA_DIR / "extended_games.csv")


def load_item_reviews() -> pd.DataFrame:
    """
    Load aggregated review statistics per item.

    Returns
    -------
    pd.DataFrame
        item_reviews.csv, one row per item_id.
    """
    return pd.read_csv(DATA_DIR / "item_reviews.csv")


def load_user_reviews() -> pd.DataFrame:
    """
    Load aggregated review statistics per user.

    Returns
    -------
    pd.DataFrame
        user_reviews.csv, one row per user_id.
    """
    return pd.read_csv(DATA_DIR / "user_reviews.csv")


def load_bundles() -> pd.DataFrame:
    """
    Load bundle to game relationships.

    Returns
    -------
    pd.DataFrame
        bundles.csv, describing which games belong to which bundle.
    """
    return pd.read_csv(DATA_DIR / "bundles.csv")