import pandas as pd

def rerank_with_novelty(recs_df, test_in, item_distance, lambda_val=0.2):
    """
    Score update: s' = s + lambda * novelty(u,i).
    novelty(u,i) = average distance between item i and all items user u has seen.
    """
    # user_id -> list of seen item_ids
    user_history = test_in.groupby("user_id")["item_id"].apply(list)

    rows = []

    for u, i, s in recs_df[["user_id", "item_id", "score"]].itertuples(index=False):
        history_items = user_history[u]

        # novelty(u,i)
        novelty_val = item_distance[i, history_items].mean()

        new_score = s + lambda_val * novelty_val
        rows.append((u, i, new_score))

    return pd.DataFrame(rows, columns=["user_id", "item_id", "score"])