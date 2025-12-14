import numpy as np
import pandas as pd

from src.data.mappings import build_id_mappings
from src.data.matrix import build_user_item_matrix

from scipy.sparse import csr_matrix

def build_user_histories(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    Build a mapping user_id -> array of item_ids from the given interactions df.

    Assumes df has at least ['user_id', 'item_id'].
    """
    user_hist = (
        df.groupby("user_id")["item_id"]
        .apply(lambda s: s.to_numpy(dtype=np.int64))
        .to_dict()
    )
    return user_hist


def build_interaction_matrix(interactions: pd.DataFrame) -> csr_matrix:
    """
    Build a binary user item interaction matrix X of shape (n_users, n_items).

    Prints a lot of sanity info so we can see what is going on.
    """
    user_ids = interactions["user_id"].astype(np.int64).values
    item_ids = interactions["item_id"].astype(np.int64).values

    n_users_matrix = user_ids.max() + 1
    n_items_matrix = item_ids.max() + 1

    # Binary implicit feedback
    data = np.ones(len(interactions), dtype=np.float32)
    X = csr_matrix(
        (data, (user_ids, item_ids)),
        shape=(n_users_matrix, n_items_matrix),
    )
    return X


def generate_recommendations(model,
                             train_df: pd.DataFrame,
                             test_in_df: pd.DataFrame,
                             top_k: int = 20) -> pd.DataFrame:
    """
    Generate top k recommendations for each user in test_in_df.

    Training:
      - EASE is fit on train_df only.

    Scoring:
      - User histories are built from test_in_df (fold-in interactions).

    Offline:
      - You call recommend(train_in, train_in, ...) so both matrices are the same.
    Online (Codabench):
      - You call recommend(train_full, test_in, ...) so training and histories differ.
    """

    # 1. Build training matrix and fit EASE
    X_train = build_interaction_matrix(train_df)

    if hasattr(model, "fit") and getattr(model, "W", None) is None:
        model.fit(X_train)

    n_items = X_train.shape[1]

    # 2. Build user histories from test_in_df
    user_hist = build_user_histories(test_in_df)

    # 3. Users we need to score are exactly those in test_in_df
    target_users = np.sort(test_in_df["user_id"].unique())

    recs = []

    for idx, u in enumerate(target_users):
        # Build user_vec from fold-in history only
        user_vec = np.zeros(n_items, dtype=np.float32)

        items_u = user_hist.get(u, None)
        if items_u is not None:
            # Guard in case some item_id is outside the training range
            valid_items = items_u[items_u < n_items]
            user_vec[valid_items] = 1.0

        scores = model.predict_user(user_vec)

        # mask history
        history_idx = user_vec.nonzero()[0]
        scores[history_idx] = -np.inf

        # top k ranking
        if top_k >= len(scores):
            ranked_items = np.argsort(-scores)
        else:
            candidate_idx = np.argpartition(-scores, top_k)[:top_k]
            ranked_rel = np.argsort(-scores[candidate_idx])
            ranked_items = candidate_idx[ranked_rel]

        for rank, item_id in enumerate(ranked_items[:top_k], start=1):
            recs.append(
                {
                    "user_id": int(u),
                    "item_id": int(item_id),
                    "score": float(scores[item_id]),
                    "rank": rank,
                }
            )
    recs = pd.DataFrame(recs)

    return recs