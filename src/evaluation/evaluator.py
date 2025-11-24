import pandas as pd
from src.evaluation.metrics import (
    calculate_ndcg,
    calculate_calibrated_recall,
    calculate_user_coverage,
    calculate_item_coverage,
    calculate_item_gini,
    calculate_publisher_gini,
    calculate_intra_list_sim,
    calculate_novelty,
    get_top_k,
)

def evaluate_model(
    recs: pd.DataFrame,
    train_in: pd.DataFrame,
    test_out: pd.DataFrame,
    publisher_mapper=None,
    item_similarity=None,
    item_distance=None,
    k=20
):
    """
    Codabench-aligned evaluation metrics.
    Mirrors exactly the Codabench leaderboard.
    """

    # enforce top-k as Codabench does
    recs_k = get_top_k(recs, k)

    n_users = train_in["user_id"].nunique()
    n_items = train_in["item_id"].nunique()

    # Build metrics
    results = {}

    # Accuracy
    results["ndcg"]   = calculate_ndcg(recs_k, k, test_out)
    results["recall"] = calculate_calibrated_recall(recs_k, k, test_out)

    # Fairness
    results["user_coverage"] = calculate_user_coverage(recs_k, k, n_users)
    results["item_gini"]     = calculate_item_gini(recs_k, k)

    if publisher_mapper is not None:
        results["publisher_gini"] = calculate_publisher_gini(
            recs_k, k, publisher_mapper
        )
    else:
        results["publisher_gini"] = None

    # Diversity
    results["item_coverage"] = calculate_item_coverage(recs_k, k, n_items)

    if item_similarity is not None:
        results["intra_list_similarity"] = calculate_intra_list_sim(
            recs_k, k, item_similarity
        )
    else:
        results["intra_list_similarity"] = None

    # Novelty
    if item_distance is not None:
        results["novelty"] = calculate_novelty(
            recs_k, k, train_in, item_distance
        )
    else:
        results["novelty"] = None

    return results
