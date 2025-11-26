import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from src.models.ease import EASE
from src.data.loader import load_interactions
from src.data.mappings import build_id_mappings
from src.data.matrix import build_user_item_matrix
from src.evaluation.metrics import (
    calculate_ndcg,
    calculate_calibrated_recall as calculate_recall,
    calculate_user_coverage,
    calculate_item_coverage,
    calculate_item_gini
)

def create_train_in_out(train_df, fold_in_ratio=0.8):
    print("[DEBUG] Creating fold-in/out split...")
    grouped = train_df.groupby("user_id")["item_id"].apply(list)

    train_in_rows = []
    train_out_rows = []

    for user, items in grouped.items():
        items = np.array(items)
        n = len(items)
        if n == 1:
            train_in_rows.append([user, items[0]])
            continue

        perm = np.random.permutation(n)
        k = int(n * fold_in_ratio)

        fold_in_items = items[perm[:k]]
        fold_out_items = items[perm[k:]]

        for it in fold_in_items:
            train_in_rows.append([user, it])
        for it in fold_out_items:
            train_out_rows.append([user, it])

    train_in = pd.DataFrame(train_in_rows, columns=["user_id", "item_id"])
    train_out = pd.DataFrame(train_out_rows, columns=["user_id", "item_id"])
    print(f"[DEBUG] train_in size: {len(train_in)}, train_out size: {len(train_out)}")
    return train_in, train_out

# -----------------------------------------------------
# Evaluation helper
# -----------------------------------------------------
def evaluate(model, train_in, train_out):
    print("[DEBUG] Building ID mappings...")
    t0 = time.time()
    user2idx, idx2user, item2idx, idx2item = build_id_mappings(train_in, train_out)
    print(f"[DEBUG] ID mappings built in {time.time() - t0:.2f}s")

    print("[DEBUG] Building CSR matrix...")
    t0 = time.time()
    X_train = build_user_item_matrix(train_in, user2idx, item2idx)
    print(f"[DEBUG] CSR built: shape = {X_train.shape}, nnz = {X_train.nnz}")
    print(f"[DEBUG] CSR build time: {time.time() - t0:.2f}s")

    print("[DEBUG] Removing zero columns...")
    t0 = time.time()
    nonzero_items = X_train.getnnz(axis=0) > 0
    print(f"[DEBUG] Zero columns: {(~nonzero_items).sum()}")

    X_train = X_train[:, nonzero_items]
    idx2item = [it for it, keep in zip(idx2item, nonzero_items) if keep]
    item2idx = {it: i for i, it in enumerate(idx2item)}
    print(f"[DEBUG] After filtering: X_train shape = {X_train.shape}")
    print(f"[DEBUG] Zero-column filtering time: {time.time() - t0:.2f}s")

    print("[DEBUG] Fitting EASE...")
    t0 = time.time()
    model.fit(X_train)
    print(f"[DEBUG] EASE.fit() completed in {time.time() - t0:.2f}s")

    print("[DEBUG] Predicting for each test user...")
    t0 = time.time()

    rows = []
    test_users = train_out["user_id"].unique()

    for user in test_users:
        u_idx = user2idx[user]
        user_vec = X_train[u_idx].toarray().ravel()

        scores = model.predict_user(user_vec)
        seen = user_vec.nonzero()[0]
        scores[seen] = -np.inf

        topk = np.argpartition(scores, -20)[-20:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        for it_idx in topk:
            rows.append([user, idx2item[it_idx]])

    recs = pd.DataFrame(rows, columns=["user_id", "item_id"])
    print(f"[DEBUG] Prediction step completed in {time.time() - t0:.2f}s")

    print("[DEBUG] Computing metrics...")
    t0 = time.time()

    n_users = len(test_users)
    n_items = len(item2idx)

    metrics = {
        "ndcg": calculate_ndcg(recs, 20, train_out),
        "recall": calculate_recall(recs, 20, train_out),
        "user_coverage": calculate_user_coverage(recs, 20, n_users),
        "item_coverage": calculate_item_coverage(recs, 20, n_items),
        "item_gini": calculate_item_gini(recs, 20),
        "publisher_gini": np.nan,
    }
    print(f"[DEBUG] Metrics computed in {time.time() - t0:.2f}s")

    return metrics

# -----------------------------------------------------
# Main sweep function
# -----------------------------------------------------
def run_ease_lambda_sweep(
        lambdas=None,
        output_csv="ease_lambda_sweep.csv",
        fold_in_ratio=0.8,
        max_users=3000,
        seed=42
):
    """
    Run a λ-sweep for EASE with fast offline evaluation.
    Includes user subsampling to keep runtime manageable.
    """

    np.random.seed(seed)

    print("[DEBUG] Loading full training interactions...")
    train_full = load_interactions(train=True)
    print(f"[DEBUG] Loaded {len(train_full)} interactions.")

    print("[DEBUG] Creating fold-in/out split...")
    train_in, train_out = create_train_in_out(train_full, fold_in_ratio)
    print(f"[DEBUG] train_in size: {len(train_in)}, train_out size: {len(train_out)}")

    # ----------------------------------------------------------------------
    # Subsample users for much faster sweeps
    # ----------------------------------------------------------------------
    users = train_out["user_id"].unique()
    print(f"[DEBUG] Total users in train_out: {len(users)}")

    if len(users) > max_users:
        print(f"[DEBUG] Subsampling {max_users} users for sweep...")
        sampled_users = np.random.choice(users, max_users, replace=False)

        train_out = train_out[train_out["user_id"].isin(sampled_users)]
        train_in = train_in[train_in["user_id"].isin(sampled_users)]
    else:
        print("[DEBUG] No subsampling needed.")

    print(f"[DEBUG] Users used for sweep: {train_out['user_id'].nunique()}")

    # ----------------------------------------------------------------------
    # Define lambda grid if none provided
    # ----------------------------------------------------------------------
    if lambdas is None:
        lambdas = [5, 10, 25, 50, 75, 100] + list(range(150, 1001, 50))

    print("[DEBUG] Lambda grid:", lambdas)

    # ----------------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------------
    print("[DEBUG] Starting lambda sweep...")
    results = []

    for lam in tqdm(lambdas, desc="Sweeping lambda"):

        print(f"\n===== λ = {lam} =====")

        model = EASE(lambda_reg=float(lam))
        metrics = evaluate(model, train_in, train_out)

        metrics["lambda"] = lam
        results.append(metrics)

    # ----------------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[DEBUG] Saved sweep results to: {output_csv}")

    return df