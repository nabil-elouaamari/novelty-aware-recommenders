import numpy as np
import pandas as pd
from pathlib import Path

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


def user_specific_rerank(recs_df, train_in, item_distance, user_lambda_map):
    """
    Per-user novelty weighting:
      final_score(u,i) = s + lambda_u * novelty(u,i)
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
    grouping="profile",   # "profile", "genre", "combined"
    ease_lambda_reg=LAMBDA_REG,
    top_k_eval=TOP_K,
    top_k_base=100,
    n_eval_users=N_EVAL_USERS,
    seed=SEED,
    output_csv=None,
):
    """
    Evaluate user-specific novelty weighting.
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

    item_similarity = np.load("../data/processed/genre_similarity.npy")
    item_distance = np.load("../data/processed/genre_distance.npy")

    # EASE
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
        raise ValueError("grouping must be profile | genre | combined")

    # rerank with user-specific weights
    recs_us = user_specific_rerank(
        recs_base,
        train_in,
        item_distance,
        user_lambda,
    )

    recs_us = recs_us.sort_values(
        ["user_id", "score"], ascending=[True, False]
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
