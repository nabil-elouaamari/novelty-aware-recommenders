"""
Simple preprocessing utilities for interaction data.

These helpers are used to clean the interaction table before building matrices or running experiments.
They stay intentionally small and composable, so they can be reused in notebooks and tests.
"""

import pandas as pd


def filter_interactions(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """
    Remove users and items with very few interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction data with at least ['user_id', 'item_id'].
    min_user_interactions : int, default 5
        Minimum number of interactions a user must have to be kept.
    min_item_interactions : int, default 5
        Minimum number of interactions an item must have to be kept.

    Returns
    -------
    pd.DataFrame
        Filtered interactions with sparse users/items removed.
    """
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()

    df = df[df["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
    df = df[df["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)]

    return df


def encode_ids(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """
    Convert user and item identifiers to dense integer indices.

    This is mainly used in unit tests and small experiments that
    require compact 0..N-1 encodings for matrix operations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with user and item columns.
    user_col : str, default "user_id"
        Name of the user column.
    item_col : str, default "item_id"
        Name of the item column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame object with user_col and item_col replaced by
        integer codes in [0, n_users) and [0, n_items).
    """
    df[user_col] = df[user_col].astype("category").cat.codes
    df[item_col] = df[item_col].astype("category").cat.codes
    return df
