import numpy as np
import pandas as pd

from src.data.mappings import build_id_mappings
from src.data.matrix import build_user_item_matrix


def generate_recommendations(model, train_df, test_in_df, top_k=20):
    user2idx, idx2user, item2idx, idx2item = build_id_mappings(train_df, test_in_df)

    X_train = build_user_item_matrix(train_df, user2idx, item2idx)
    X_test_in = build_user_item_matrix(test_in_df, user2idx, item2idx)

    model.fit(X_train)

    all_scores = X_test_in @ model.W   # shape: (n_users, n_items)

    # mask seen interactions
    X_seen = X_test_in.toarray()
    all_scores[X_seen == 1] = -np.inf

    results = []
    test_users = test_in_df["user_id"].unique()

    for user_id in test_users:
        u_idx = user2idx[user_id]   # ✔ correct row
        scores_u = all_scores[u_idx]

        top_items = np.argpartition(scores_u, -top_k)[-top_k:]
        top_items = top_items[np.argsort(scores_u[top_items])[::-1]]

        for item_idx in top_items:
            results.append({
                "user_id": user_id,
                "item_id": idx2item[item_idx],
                "score": float(scores_u[item_idx])
            })

    return pd.DataFrame(results)
