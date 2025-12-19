import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


def build_user_histories(
    df: pd.DataFrame,
    min_playtime: int | None = None,
) -> dict[int, np.ndarray]:
    """
    Build a mapping user_id -> array of item_ids from an interactions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
          - "user_id"
          - "item_id"
        Optionally:
          - "playtime" (used when min_playtime is set)

    min_playtime : int or None, default None
        If not None and "playtime" exists, only interactions with
        playtime >= min_playtime are included in the user history.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping each user_id to a NumPy array of item_ids
        that are considered part of their history.
    """
    if (min_playtime is not None) and ("playtime" in df.columns):
        df = df[df["playtime"] >= min_playtime]

    user_hist = (
        df.groupby("user_id")["item_id"]
        .apply(lambda s: s.to_numpy(dtype=np.int64))
        .to_dict()
    )
    return user_hist


def build_interaction_matrix(interactions: pd.DataFrame) -> csr_matrix:
    """
    Build a binary user item interaction matrix X of shape (n_users, n_items).

    Parameters
    ----------
    interactions : pd.DataFrame
        Must contain:
          - "user_id"
          - "item_id"
        IDs are assumed to be zero based integers.

    Returns
    -------
    csr_matrix
        Sparse matrix X where X[u, i] = 1 if user u interacted with item i
        and 0 otherwise.

    Notes
    -----
    The matrix shape is inferred from the maximum user_id and item_id:
      n_users = max(user_id) + 1
      n_items = max(item_id) + 1
    """
    user_ids = interactions["user_id"].astype(np.int64).values
    item_ids = interactions["item_id"].astype(np.int64).values

    n_users_matrix = user_ids.max() + 1
    n_items_matrix = item_ids.max() + 1

    data = np.ones(len(interactions), dtype=np.float32)
    X = csr_matrix(
        (data, (user_ids, item_ids)),
        shape=(n_users_matrix, n_items_matrix),
    )
    return X


def generate_recommendations(
    model,
    train_df: pd.DataFrame,
    test_in_df: pd.DataFrame,
    top_k: int = 20,
    min_playtime: int | None = None,
) -> pd.DataFrame:
    """
    Generate top k recommendations for each user in test_in_df.

    Training
    --------
    - The model is fitted on train_df (optionally filtered by min_playtime).

    Scoring
    -------
    - User histories are built from test_in_df and then passed to the model
      through predict_user.

    Typical usage
    -------------
    Offline evaluation:
      recommend(train_in, train_in, top_k)

    Codabench style online evaluation:
      recommend(train_full, test_in, top_k)

    Parameters
    ----------
    model :
        Any object that implements:
          - fit(X: csr_matrix)
          - predict_user(user_vector: np.ndarray) -> np.ndarray

    train_df : pd.DataFrame
        Training interactions. Must contain "user_id" and "item_id".
        It may also contain "playtime" when min_playtime is used.

    test_in_df : pd.DataFrame
        Fold in interactions that define which users to score and what
        their history is. Must contain "user_id" and "item_id".

    top_k : int, default 20
        Number of items to recommend per user.

    min_playtime : int or None, default None
        Optional filter that removes very short play sessions from both
        the training matrix and the user histories.

    Returns
    -------
    pd.DataFrame
        Recommendations with columns:
          - "user_id"
          - "item_id"
          - "score"
          - "rank"
        where rank is 1 for the top recommendation.
    """
    # optionally filter training data for the matrix
    train_for_matrix = train_df
    if (min_playtime is not None) and ("playtime" in train_for_matrix.columns):
        train_for_matrix = train_for_matrix[train_for_matrix["playtime"] >= min_playtime]

    # 1. Build training matrix and fit the model
    X_train = build_interaction_matrix(train_for_matrix)

    if hasattr(model, "fit") and getattr(model, "W", None) is None:
        model.fit(X_train)

    n_items = X_train.shape[1]

    # 2. Build user histories from test_in_df
    user_hist = build_user_histories(test_in_df, min_playtime=min_playtime)

    # 3. Users to score are exactly those present in test_in_df
    target_users = np.sort(test_in_df["user_id"].unique())

    recs = []

    for _, u in enumerate(target_users):
        # build user vector from fold in history only
        user_vec = np.zeros(n_items, dtype=np.float32)

        items_u = user_hist.get(u, None)
        if items_u is not None:
            # guard against item_ids that might not appear in the training matrix
            valid_items = items_u[items_u < n_items]
            user_vec[valid_items] = 1.0

        scores = model.predict_user(user_vec)

        # mask items already in the history
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

    recs_df = pd.DataFrame(recs)
    return recs_df
