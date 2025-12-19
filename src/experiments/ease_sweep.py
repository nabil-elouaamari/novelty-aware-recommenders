"""
Offline lambda sweep for the EASE model.

This script uses a Codabench style split where each user has one held-out interaction.
For a list of candidate lambdas it:

  - Fits EASE on train_in
  - generates recommendations on train_in
  - evaluates NDCG, recall, and other metrics with evaluate_model
  - Saves results as a CSV under notebooks/results/sweeps
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.models.ease import EASE
from src.data.loader import load_interactions
from src.evaluation.splitter import split_train_in_out
from src.evaluation.evaluator import evaluate_model
from src.config import N_EVAL_USERS, SEED, TOP_K, POP_ALPHA


def _prepare_offline_split(
    max_users: int = N_EVAL_USERS,
    seed: int = SEED,
):
    """
    Prepare an offline train_in / train_out split similar to the
    protocol used in the baseline notebook.

    Steps
    -----
      1) Load full train_interactions.
      2) Apply split_train_in_out to get one held-out item per user.
      3) Subsample up to max_users users with a holdout for faster runs.

    Parameters
    ----------
    max_users : int, default N_EVAL_USERS
        Maximum number of users to keep for the sweep.
    seed : int, default SEED
        Random seed for reproducible user sampling.

    Returns
    -------
    train_in : pd.DataFrame
        Fold in interactions.
    train_out : pd.DataFrame
        Held out interactions per sampled user.
    """
    # full training interactions
    train_full = load_interactions(train=True)

    # Codabench style split: one holdout per user
    train_in_full, train_out_full = split_train_in_out(train_full, seed=seed)

    # subsample users with a holdout
    rng = np.random.default_rng(seed)
    all_users = train_out_full["user_id"].unique()
    n_eval = min(max_users, len(all_users))
    sampled_users = rng.choice(all_users, size=n_eval, replace=False)

    train_in = (
        train_in_full[train_in_full["user_id"].isin(sampled_users)]
        .reset_index(drop=True)
    )
    train_out = (
        train_out_full[train_out_full["user_id"].isin(sampled_users)]
        .reset_index(drop=True)
    )

    return train_in, train_out


def run_ease_lambda_sweep(
    lambdas: List[float],
    output_csv: str = "ease_lambda_sweep.csv",
    max_users: int = N_EVAL_USERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Run a lambda sweep for EASE using the offline protocol from
    _prepare_offline_split.

    For each lambda:
      - instantiate EASE(lambda_reg=lambda, alpha_pop=POP_ALPHA)
      - generate recommendations on (train_in, train_in)
      - Evaluate with evaluate_model (ndcg, recall, coverage, gini, ...)

    Parameters
    ----------
    lambdas : list of float
        Candidate regularization values to evaluate.
    output_csv : str, default "ease_lambda_sweep.csv"
        File name for the saved CSV under notebooks/results/sweeps.
    max_users : int, default N_EVAL_USERS
        Maximum number of users to include in the evaluation split.
    seed : int, default SEED
        Random seed passed to the splitter.

    Returns
    -------
    pd.DataFrame
        One row per lambda with all computed metrics.
    """
    if not lambdas:
        raise ValueError("lambdas must be a non empty list, for example [50, 100, 200].")

    # prepare to split once
    train_in, train_out = _prepare_offline_split(
        max_users=max_users,
        seed=seed,
    )

    results: list[dict] = []

    print(f"[INFO] Running EASE sweep with {len(lambdas)} lambdas...")
    for lam in lambdas:
        print(
            "[INFO] Evaluating lambda =",
            lam,
            "| progress:",
            round(len(results) / len(lambdas) * 100, 2),
            "%",
        )

        # 1) model with current lambda
        model = EASE(lambda_reg=float(lam), alpha_pop=float(POP_ALPHA))

        # 2) recommendations
        recs = model.recommend(
            train_in,
            train_in,
            top_k=TOP_K,
        )

        # 3) evaluation
        metrics = evaluate_model(
            recs,
            train_in,
            train_out,
            publisher_mapper=None,  # skip publisher gini for speed
            item_similarity=None,   # skip similarity-based metrics
            item_distance=None,     # skip novelty here
            k=TOP_K,
        )

        metrics["lambda"] = lam
        results.append(metrics)

    # aggregate and save
    df = pd.DataFrame(results)

    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "notebooks" / "results" / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_csv
    df.to_csv(output_path, index=False)

    return df
