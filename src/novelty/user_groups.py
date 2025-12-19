import numpy as np
import pandas as pd
import ast
from collections import Counter


# ============================================================
# 1) PROFILE SIZE GROUPING (based on interaction count)
# ============================================================

def assign_groups_by_profile_size(
    train_in: pd.DataFrame,
    small: int = 15,
    medium: int = 50,
) -> dict[int, float]:
    """
    Assign a novelty weight per user based on profile size.

    Parameters
    ----------
    train_in : pd.DataFrame
        Interactions used as fold in data. Must contain:
          - "user_id"
          - "item_id"

    small : int, default 15
        Threshold for "small" profiles. Users with fewer than this
        number of interactions receive a low novelty weight.

    medium : int, default 50
        Threshold for "medium" profiles. Users with fewer than this
        but at least "small" interactions receive a medium novelty weight.
        Users with at least "medium" interactions are treated as large profiles.

    Returns
    -------
    dict[int, float]
        Mapping user_id -> novelty_weight, using:
          - small profile  (count < small)    -> lambda = 0.10
          - medium profile (count < medium)   -> lambda = 0.20
          - large profile  (count >= medium)  -> lambda = 0.30

    Notes
    -----
    The exact lambda values and thresholds are hand tuned and chosen
    for simplicity rather than optimality.
    """
    counts = train_in["user_id"].value_counts()

    user_lambda: dict[int, float] = {}

    for user, c in counts.items():
        if c < small:
            user_lambda[user] = 0.10
        elif c < medium:
            user_lambda[user] = 0.20
        else:
            user_lambda[user] = 0.30

    return user_lambda


# ============================================================
# 2) GENRE DIVERSITY (entropy)
# ============================================================

def compute_genre_entropy(
    history_items: list[int],
    item_genres: dict[int, list[str]],
) -> float:
    """
    Compute the entropy of genres in a user's history.

    Parameters
    ----------
    history_items : list[int]
        List of item_ids the user has interacted with.

    item_genres : dict[int, list[str]]
        Mapping item_id -> list of genre strings.

    Returns
    -------
    float
        Shannon entropy of the genre distribution.
        Zero if the user has no genres or an empty history.
    """
    genres: list[str] = []
    for it in history_items:
        genres.extend(item_genres.get(it, []))

    if len(genres) == 0:
        return 0.0

    freq = Counter(genres)
    total = sum(freq.values())
    probs = np.array(list(freq.values())) / total
    entropy = -(probs * np.log(probs + 1e-12)).sum()

    return entropy


def assign_groups_by_genre_diversity(
    train_in: pd.DataFrame,
    games_df: pd.DataFrame,
) -> dict[int, float]:
    """
    Assign novelty weights based on genre diversity of user histories.

    Parameters
    ----------
    train_in : pd.DataFrame
        Interactions used as fold in data. Must contain:
          - "user_id"
          - "item_id"

    games_df : pd.DataFrame
        Game metadata. Must contain:
          - "item_id"
          - "genres" as a stringified list

    Returns
    -------
    dict[int, float]
        Mapping user_id -> novelty_weight where users are grouped by
        the entropy of their genre distribution:
          - low entropy    -> lambda = 0.10
          - medium entropy -> lambda = 0.20
          - high entropy   -> lambda = 0.30

    Notes
    -----
    Entropy thresholds are defined by the 33rd and 66th percentiles
    of the entropy distribution across users.
    """
    # Build item_id -> genres list
    item_genres: dict[int, list[str]] = {}
    for _, row in games_df.iterrows():
        try:
            item_genres[row["item_id"]] = ast.literal_eval(row["genres"])
        except Exception:
            item_genres[row["item_id"]] = []

    user_history = train_in.groupby("user_id")["item_id"].apply(list)

    # compute entropy per user
    entropies: dict[int, float] = {}
    for user, hist_items in user_history.items():
        entropies[user] = compute_genre_entropy(hist_items, item_genres)

    values = np.array(list(entropies.values()))
    q1, q2 = np.quantile(values, [0.33, 0.66])

    user_lambda: dict[int, float] = {}

    for user, H in entropies.items():
        if H < q1:
            user_lambda[user] = 0.10
        elif H < q2:
            user_lambda[user] = 0.20
        else:
            user_lambda[user] = 0.30

    return user_lambda


# ============================================================
# 3) OPTIONAL COMBINED STRATEGY
# ============================================================

def assign_groups_combined(
    train_in: pd.DataFrame,
    games_df: pd.DataFrame,
) -> dict[int, float]:
    """
    Combine profile size and genre diversity into a single novelty weight.

    Parameters
    ----------
    train_in : pd.DataFrame
        Interactions used as fold in data. Must contain "user_id" and "item_id".

    games_df : pd.DataFrame
        Game metadata as in assign_groups_by_genre_diversity.

    Returns
    -------
    dict[int, float]
        Mapping user_id -> novelty_weight where:
          lambda_combined = average(lambda_profile, lambda_genre)
        with both lambdas taken from the two grouping strategies above.

    Notes
    -----
    This function is a simple heuristic. It does not learn weights from data,
    it only averages the two hand designed schemes.
    """
    lam_profile = assign_groups_by_profile_size(train_in)
    lam_genre = assign_groups_by_genre_diversity(train_in, games_df)

    user_lambda: dict[int, float] = {}
    for user in lam_profile:
        user_lambda[user] = (lam_profile[user] + lam_genre[user]) / 2.0

    return user_lambda