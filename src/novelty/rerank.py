import pandas as pd
import numpy as np


def rerank_with_novelty(
    recs_df: pd.DataFrame,
    test_in: pd.DataFrame,
    item_distance: np.ndarray,
    lambda_val: float = 0.2,
) -> pd.DataFrame:
    """
    Apply global novelty-based reranking to a recommendation list.

    Score update:
        s'(u, i) = s(u, i) + lambda_val * novelty(u, i)

    where:
        novelty(u, i) = average distance between candidate item i and
                        all items in user u's history, based on item_distance.

    Parameters
    ----------
    recs_df : pd.DataFrame
        Base recommendations. Must contain columns:
          - "user_id"
          - "item_id"
          - "score" (EASE or EASE + popularity score)

    test_in : pd.DataFrame
        Fold in interactions for the same users. Must contain:
          - "user_id"
          - "item_id"
        This defines the history used to compute novelty.

    item_distance : np.ndarray
        Precomputed item distance matrix of shape (n_items, n_items),
        typically built from genres with build_genre_distance_matrix.

    lambda_val : float, default 0.2
        Global weight for novelty. Higher values increase the impact of
        novelty(u, i) relative to the original score.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["user_id", "item_id", "score"] where
        score is the updated score s'(u, i). Rows are not sorted, so
        callers usually sort by ["user_id", "score"] in a later step.
    """
    # user_id -> list of seen item_ids
    user_history = test_in.groupby("user_id")["item_id"].apply(list)

    rows = []

    for u, i, s in recs_df[["user_id", "item_id", "score"]].itertuples(index=False):
        history_items = user_history[u]

        # novelty(u, i) = average distance from item i to all items in the history
        novelty_val = item_distance[i, history_items].mean()

        new_score = s + lambda_val * novelty_val
        rows.append((u, i, new_score))

    return pd.DataFrame(rows, columns=["user_id", "item_id", "score"])