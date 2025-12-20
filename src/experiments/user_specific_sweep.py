"""
User-specific novelty experiments.

This script compares three simple strategies for assigning per user
novelty weights:

  - profile: based on profile size (number of interactions)
  - genre: based on genre entropy in the user history
  - combined: average of profile-based and genre-based weights

For each strategy, it reranks EASE recommendations with:
  final_score(u, i) = s_ease(u, i) + lambda_u * novelty(u, i)

And evaluates accuracy, novelty, and diversity metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_interactions, load_games
from src.evaluation.splitter import split_train_in_out
from src.evaluation.evaluator import evaluate_model
from src.models.ease import EASE
from src.novelty.user_groups import (
    assign_groups_by_profile_size,
    assign_groups_by_genre_diversity,
    assign_groups_combined,
)
from src.config import LAMBDA_REG, TOP_K, N_EVAL_USERS, SEED


def user_specific_rerank(
    recs_df: pd.DataFrame,
    train_in: pd.DataFrame,
    item_distance: np.ndarray,
    user_lambda_map: dict,
) -> pd.DataFrame:
    """
    Apply per user novelty weighting to a base recommendation list.

    The new score is
        final_score(u, i) = s + lambda_u * novelty(u, i)

    Where novelty(u, i) is the average distance between candidate item i
    and all items in user u's history.

    Parameters
    ----------
    recs_df : pd.DataFrame
        Base recommendation list with columns
        ['user_id', 'item_id', 'score'].
    train_in : pd.DataFrame
        Fold in interactions, used to build user histories.
    item_distance : np.ndarray
        Item item distance matrix indexed by item_id.
    user_lambda_map : dict
        Mapping user_id -> novelty weight lambda_u.

    Returns
    -------
    pd.DataFrame
        Reranked recommendations with updated 'score' values.
    """
    user_hist = train_in.groupby("user_id")["item_id"].apply(list)
    rows = []

    for u, i, s in recs_df[["user_id", "item_id", "score"]].itertuples(index=False):
        lam_u = user_lambda_map[u]
        novelty_val = item_distance[i, user_hist[u]].mean()
        new_score = s + lam_u * novelty_val
        rows.append((u, i, new_score))

    return pd.DataFrame(rows, columns=["user_id", "item_id", "score"])


def run_user_specific_sweep(
    grouping: str = "profile",   # "profile", "genre", "combined"
    ease_lambda_reg: float = LAMBDA_REG,
    top_k_eval: int = TOP_K,
    top_k_base: int = 100,
    n_eval_users: int = N_EVAL_USERS,
    seed: int = SEED,
    output_csv: str | None = None,
) -> dict:
    """
    Evaluate a user-specific novelty weighting strategy.

    Steps
    -----
      1) Load full train_interactions and games.
      2) Split interactions into train_in / train_out.
      3) Sample up to n_eval_users users.
      4) Train EASE on train_in and generate a base list of length
         top_k_base per user.
      5) Assign per user novelty weights lambda_u using one of:
           - assign_groups_by_profile_size
           - assign_groups_by_genre_diversity
           - assign_groups_combined
      6) Apply user_specific_rerank to adjust scores.
      7) Sort and evaluate with evaluate_model.
      8) Save metrics as a one-row CSV.

    Parameters
    ----------
    grouping: {"profile", "genre", "combined"}, default "profile"
        Which grouping strategy to use to assign lambda_u.
    ease_lambda_reg : float, default LAMBDA_REG
        Regularization value for the underlying EASE model.
    top_k_eval : int, default TOP_K
        Cutoff k for evaluation metrics.
    top_k_base : int, default 100
        Length of the base recommendation list before reranking.
    n_eval_users : int, default N_EVAL_USERS
        Maximum number of users in the offline split.
    seed : int, default SEED
        Random seed for splitting and sampling.
    output_csv : str or None
        Output file name under notebooks/results/sweeps. If None, a
        default name user_specific_<grouping>.csv is used.

    Returns
    -------
    dict
        Dictionary of metrics as returned by evaluate_model, extended
        with keys:
          - grouping
          - ease_lambda_reg
    """
    rng = np.random.default_rng(seed)

    train = load_interactions(train=True)
    games = load_games()
    publisher_mapper = games.set_index("item_id")["publisher"]

    # split
    train_in_full, train_out_full = split_train_in_out(train, seed=seed)

    all_users = train_out_full["user_id"].unique()
    n_eval = min(n_eval_users, len(all_users))
    sample_users = rng.choice(all_users, size=n_eval, replace=False)

    train_in = train_in_full[train_in_full["user_id"].isin(sample_users)]
    train_out = train_out_full[train_out_full["user_id"].isin(sample_users)]

    item_similarity = np.load("results/processed/genre_similarity.npy")
    item_distance = np.load("results/processed/genre_distance.npy")

    # EASE baseline
    model = EASE(lambda_reg=ease_lambda_reg)
    recs_base = model.recommend(train_in, train_in, top_k=top_k_base)

    # choose grouping
    if grouping == "profile":
        user_lambda = assign_groups_by_profile_size(train_in)
    elif grouping == "genre":
        user_lambda = assign_groups_by_genre_diversity(train_in, games)
    elif grouping == "combined":
        user_lambda = assign_groups_combined(train_in, games)
    else:
        raise ValueError("grouping must be 'profile', 'genre' or 'combined'")

    # rerank with user-specific weights
    recs_us = user_specific_rerank(
        recs_base,
        train_in,
        item_distance,
        user_lambda,
    )

    recs_us = recs_us.sort_values(
        ["user_id", "score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # evaluate
    metrics = evaluate_model(
        recs_us,
        train_in,
        train_out,
        publisher_mapper=publisher_mapper,
        item_similarity=item_similarity,
        item_distance=item_distance,
        k=top_k_eval,
    )

    metrics["grouping"] = grouping
    metrics["ease_lambda_reg"] = ease_lambda_reg

    # save
    if output_csv is None:
        output_csv = f"user_specific_{grouping}.csv"

    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "notebooks" / "results" / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])
    output_path = output_dir / output_csv
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved sweep submissions to: {output_path}")

    return metrics
