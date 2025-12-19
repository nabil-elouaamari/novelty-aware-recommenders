"""
High level evaluation wrapper.

This module contains a single helper `evaluate_model` that computes the
same set of metrics that Codabench uses for the official leaderboard,
plus novelty and intra list similarity when the required inputs are
provided.

It takes raw recommendation lists and interaction dataframes, applies a top k cut,
and delegates the actual metric calculations to src.evaluation.metrics.
"""

from typing import Any, Dict, Optional
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
    publisher_mapper: Optional[pd.Series] = None,
    item_similarity = None,
    item_distance = None,
    k: int = 20,
) -> Dict[str, Any]:
    """
    Evaluate a recommendation list using Codabench aligned metrics.

    Parameters
    ----------
    recs : pd.DataFrame
        Recommendation list with at least ['user_id', 'item_id'].
        A 'score' column is allowed but not required.
    train_in : pd.DataFrame
        Fold in interactions used to build the user histories.
        Used here to infer the number of users and items plus
        the histories for novelty.
    test_out : pd.DataFrame
        Held out interactions per user, one or more items per user.
        Used as the ground truth for accuracy metrics.
    publisher_mapper : pd.Series, optional
        Series indexed by item_id that maps an item to its publisher.
        Required for publisher_gini, otherwise that metric is set to None.
    item_similarity : np.ndarray or None
        Precomputed item by item similarity matrix in the order of
        item_ids. Required for intra list similarity.
    item_distance : np.ndarray or None
        Precomputed item by item distance matrix. Required for novelty.
    k : int, default 20
        Cutoff for all top-k metrics. The recommendation list will be
        truncated to the top k items per user before computing metrics.

    Returns
    -------
    dict
        Dictionary with the following keys:
          - ndcg
          - recall
          - user_coverage
          - item_coverage
          - item_gini
          - publisher_gini
          - intra_list_similarity
          - novelty
        Metrics that cannot be computed with the given inputs are
        returned as None.
    """
    # Enforce top-k cut, just like Codabench
    recs_k = get_top_k(recs, k)

    n_users = train_in["user_id"].nunique()
    n_items = train_in["item_id"].nunique()

    results: Dict[str, Any] = {}

    # Accuracy
    results["ndcg"] = calculate_ndcg(recs_k, k, test_out)
    results["recall"] = calculate_calibrated_recall(recs_k, k, test_out)

    # User and item level coverage and concentration
    results["user_coverage"] = calculate_user_coverage(recs_k, k, n_users)
    results["item_gini"] = calculate_item_gini(recs_k, k)

    if publisher_mapper is not None:
        results["publisher_gini"] = calculate_publisher_gini(
            recs_k,
            k,
            publisher_mapper,
        )
    else:
        results["publisher_gini"] = None

    # Diversity
    results["item_coverage"] = calculate_item_coverage(recs_k, k, n_items)

    if item_similarity is not None:
        results["intra_list_similarity"] = calculate_intra_list_sim(
            recs_k,
            k,
            item_similarity,
        )
    else:
        results["intra_list_similarity"] = None

    # Novelty
    if item_distance is not None:
        results["novelty"] = calculate_novelty(
            recs_k,
            k,
            train_in,
            item_distance,
        )
    else:
        results["novelty"] = None

    return results