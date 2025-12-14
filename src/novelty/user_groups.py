import numpy as np
import pandas as pd
import ast
from collections import Counter


# ============================================================
# 1) PROFILE SIZE GROUPING (based on interaction count)
# ============================================================
def assign_groups_by_profile_size(train_in, small=15, medium=50):
    """
    Returns dict: user_id -> novelty_weight
    Groups:
      - small profile (< small) -> λ = 0.10
      - medium profile (< medium) -> λ = 0.20
      - large profile (>= medium) -> λ = 0.30
    """

    counts = train_in["user_id"].value_counts()

    user_lambda = {}

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
def compute_genre_entropy(history_items, item_genres):
    """
    entropy (history genres)
    history_items: list of item_ids
    item_genres: dict item_id -> list of genres
    """
    genres = []
    for it in history_items:
        genres.extend(item_genres.get(it, []))

    if len(genres) == 0:
        return 0.0

    freq = Counter(genres)
    total = sum(freq.values())
    probs = np.array(list(freq.values())) / total
    entropy = -(probs * np.log(probs + 1e-12)).sum()

    return entropy


def assign_groups_by_genre_diversity(train_in, games_df):
    """
    Groups users based on genre entropy:
      low entropy -> λ = 0.10
      medium -> λ = 0.20
      high -> λ = 0.30
    """

    # Build item_id -> genres list
    item_genres = {}
    for _, row in games_df.iterrows():
        try:
            item_genres[row["item_id"]] = ast.literal_eval(row["genres"])
        except:
            item_genres[row["item_id"]] = []

    user_history = train_in.groupby("user_id")["item_id"].apply(list)

    entropies = {}
    for user, hist_items in user_history.items():
        entropies[user] = compute_genre_entropy(hist_items, item_genres)

    values = np.array(list(entropies.values()))
    q1, q2 = np.quantile(values, [0.33, 0.66])

    user_lambda = {}

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
def assign_groups_combined(train_in, games_df):
    """
    Combine profile size and genre diversity:
      novelty_weight = average(lambda_profile, lambda_genre)
    """

    lam_profile = assign_groups_by_profile_size(train_in)
    lam_genre = assign_groups_by_genre_diversity(train_in, games_df)

    user_lambda = {}
    for user in lam_profile:
        user_lambda[user] = (lam_profile[user] + lam_genre[user]) / 2.0

    return user_lambda
