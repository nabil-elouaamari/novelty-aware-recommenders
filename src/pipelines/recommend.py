import numpy as np
import pandas as pd

from src.data.mappings import build_id_mappings
from src.data.matrix import build_user_item_matrix

from scipy.sparse import csr_matrix
import time

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
    print(f"[build_user_histories] built histories for {len(user_hist)} users")
    return user_hist


def compute_item_popularity(train_df: pd.DataFrame, n_items: int) -> np.ndarray:
    """
    Compute per item popularity counts and turn them into a prior vector.

    popularity[i] = log(1 + count of interactions for item i in train_df)

    Returns an array of shape (n_items,) aligned with the columns of X.
    Items that never appear in train_df get prior 0.
    """

    counts = train_df["item_id"].value_counts()

    pop_counts = np.zeros(n_items, dtype=np.float32)
    idx = counts.index.to_numpy(dtype=np.int64)
    vals = counts.to_numpy(dtype=np.float32)

    pop_counts[idx] = vals

    # log1p is log(1 + x)
    pop_prior = np.log1p(pop_counts).astype(np.float32)

    print(
        "\n[compute_item_popularity] built popularity prior",
        f"n_items = {n_items}, min = {pop_prior.min():.4f},",
        f"max = {pop_prior.max():.4f}, mean = {pop_prior.mean():.4f}",
    )

    return pop_prior



def build_interaction_matrix(interactions: pd.DataFrame) -> csr_matrix:
    """
    Build a binary user item interaction matrix X of shape (n_users, n_items).

    Prints a lot of sanity info so we can see what is going on.
    """

    t0 = time.time()

    n_rows = len(interactions)
    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()

    print("\n[build_interaction_matrix] starting")
    print(f"  rows in interactions: {n_rows}")
    print(f"  unique users:         {n_users}")
    print(f"  unique items:         {n_items}")
    print(
        f"  user_id range:        {interactions['user_id'].min()} "
        f"to {interactions['user_id'].max()}"
    )
    print(
        f"  item_id range:        {interactions['item_id'].min()} "
        f"to {interactions['item_id'].max()}"
    )

    user_ids = interactions["user_id"].astype(np.int64).values
    item_ids = interactions["item_id"].astype(np.int64).values

    # quick checks
    if user_ids.min() != 0:
        print(
            f"  [warn] user_ids do not start at 0 (min = {user_ids.min()}). "
            "Matrix will have empty leading rows."
        )
    if item_ids.min() != 0:
        print(
            f"  [warn] item_ids do not start at 0 (min = {item_ids.min()}). "
            "Matrix will have empty leading columns."
        )

    n_users_matrix = user_ids.max() + 1
    n_items_matrix = item_ids.max() + 1

    print(f"  matrix users (rows):  {n_users_matrix}")
    print(f"  matrix items (cols):  {n_items_matrix}")

    # Binary implicit feedback
    data = np.ones(len(interactions), dtype=np.float32)
    X = csr_matrix(
        (data, (user_ids, item_ids)),
        shape=(n_users_matrix, n_items_matrix),
    )

    nnz = X.nnz
    density = nnz / (X.shape[0] * X.shape[1])
    avg_items_per_user = nnz / X.shape[0]
    avg_users_per_item = nnz / X.shape[1]

    print(f"  nnz interactions:     {nnz}")
    print(f"  density:              {density:.6f}")
    print(f"  avg items per user:   {avg_items_per_user:.2f}")
    print(f"  avg users per item:   {avg_users_per_item:.2f}")
    print(f"[build_interaction_matrix] done in {time.time() - t0:.2f} s\n")

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

    import time

    t0 = time.time()
    print("\n[generate_recommendations] starting")
    print(f"  train_df rows: {len(train_df)}")
    print(f"  test_in_df rows: {len(test_in_df)}")
    print(f"  users in train_df: {train_df['user_id'].nunique()}")
    print(f"  users in test_in_df: {test_in_df['user_id'].nunique()}")

    # 1. Build training matrix and fit EASE
    X_train = build_interaction_matrix(train_df)

    if hasattr(model, "fit") and getattr(model, "W", None) is None:
        print("  model.W is None, fitting EASE on X_train")
        model.fit(X_train)
    else:
        print("  model appears already fitted, skipping fit")

    n_items = X_train.shape[1]

    # 2. Build user histories from test_in_df
    user_hist = build_user_histories(test_in_df)

    # 3. Users we need to score are exactly those in test_in_df
    target_users = np.sort(test_in_df["user_id"].unique())
    print(f"  target users to score: {len(target_users)}")

    recs = []
    debug_user_limit = 5

    for idx, u in enumerate(target_users):
        # Build user_vec from fold-in history only
        user_vec = np.zeros(n_items, dtype=np.float32)

        items_u = user_hist.get(u, None)
        if items_u is not None:
            # Guard in case some item_id is outside training range
            valid_items = items_u[items_u < n_items]
            user_vec[valid_items] = 1.0

        # Sanity: count interactions we are using
        n_hist = int(user_vec.sum())

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

        if idx < debug_user_limit:
            preview_items = ranked_items[:5]
            print(
                f"  [user {u}] interactions (fold-in): {n_hist}, "
                f"top5 items: {list(preview_items)}"
            )
            print(
                "             top5 scores:",
                [float(scores[i]) for i in preview_items],
            )

        if idx == debug_user_limit:
            print("  ... (suppressing further per user prints)")

    recs = pd.DataFrame(recs)

    print(
        f"[generate_recommendations] produced {len(recs)} rows "
        f"in {time.time() - t0:.2f} s\n"
    )

    return recs