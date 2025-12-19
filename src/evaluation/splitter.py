"""
Train / test splitting utilities for offline evaluation.

The main helper here, split_train_in_out, mimics the Codabench setup:
for each user, exactly one interaction is held out as test_out and
all remaining interactions are used as train_in (fold-in history).
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.config import SEED


def split_train_in_out(
    interactions: pd.DataFrame,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per user random holdout split in the Codabench style.

    For each user, one interaction is randomly selected as the held out
    item (test_out). All other interactions for that user go to train_in.

    Users with only one interaction cannot have a held out item:
    for those users, their only interaction is kept in train_in and
    they do not appear in test_out.

    Parameters
    ----------
    interactions : pd.DataFrame
        Full interaction table with at least ['user_id', 'item_id'].
    seed : int, default SEED from config
        Random seed for reproducible per user sampling.

    Returns
    -------
    train_in : pd.DataFrame
        Interactions used as fold-in history.
    test_out : pd.DataFrame
        One held out interaction per eligible user.
    """
    rng = np.random.default_rng(seed)

    train_in_parts = []
    test_out_parts = []

    for _, df_user in interactions.groupby("user_id"):
        # If a user has only one interaction, everything stays in train_in
        if len(df_user) == 1:
            train_in_parts.append(df_user)
            continue

        # Randomly choose one row index to hold out
        idx = rng.choice(df_user.index, 1)
        test_row = df_user.loc[idx]
        remaining = df_user.drop(idx)

        train_in_parts.append(remaining)
        test_out_parts.append(test_row)

    train_in = pd.concat(train_in_parts).reset_index(drop=True)
    test_out = pd.concat(test_out_parts).reset_index(drop=True)

    return train_in, test_out
