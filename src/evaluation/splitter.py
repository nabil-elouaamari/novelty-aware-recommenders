import pandas as pd
import numpy as np

def split_train_in_out(interactions: pd.DataFrame, seed: int = 42):
    """
    Codabench-aligned per-user split.
    For each user, hold out exactly one interaction as test_out.
    The remaining interactions are used as test_in (fold-in).
    """
    rng = np.random.default_rng(seed)

    train_in = []
    test_out = []

    for user, df in interactions.groupby("user_id"):
        # if a user has only 1 interaction, they cannot have a test_out
        if len(df) == 1:
            train_in.append(df)
            continue

        # randomly choose 1 item to hold out
        idx = rng.choice(df.index, 1)
        test_row = df.loc[idx]
        remaining = df.drop(idx)

        train_in.append(remaining)
        test_out.append(test_row)

    train_in = pd.concat(train_in).reset_index(drop=True)
    test_out = pd.concat(test_out).reset_index(drop=True)

    return train_in, test_out
