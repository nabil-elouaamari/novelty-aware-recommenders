from pathlib import Path

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
    Prepare an offline train_in / train_out split that matches
    the logic in 02_ease_baseline.ipynb:

      - 1 holdout per user via split_train_in_out
      - subsample up to max_users users for faster evaluation
    """
    # full training interactions
    train_full = load_interactions(train=True)

    # Codabench-like split: one holdout per user
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
    lambdas: list[float],
    output_csv: str = "ease_lambda_sweep.csv",
    max_users: int = N_EVAL_USERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Run a λ-sweep for EASE using the same offline protocol as
    in 02_ease_baseline.ipynb.

    Shared setup (done once):
      - load full train_interactions
      - split into train_in / train_out (1 holdout per user)
      - Subsample up to max_users users

    Per λ:
      - instantiate EASE(lambda_reg=λ)
      - recommend on (train_in, train_in)
      - Evaluate with evaluate_model (ndcg, recall, coverage, gini, ...)

    Results are saved to results/sweeps/<output_csv> and also
    returned as a DataFrame.
    """
    if not lambdas:
        raise ValueError("lambdas must be a non-empty list, e.g. [50, 100, 200].")

    # prepare to split once
    train_in, train_out = _prepare_offline_split(
        max_users=max_users,
        seed=seed,
    )

    results: list[dict] = []

    print(f"[INFO] Running EASE sweep with {len(lambdas)} lambdas...")
    for lam in lambdas:
        print("[INFO] Evaluating lambda =", lam, "| progress:", round(len(results) / len(lambdas) * 100, 2), "%")
        # 1) model with current lambda
        model = EASE(lambda_reg=float(lam), alpha_pop=float(POP_ALPHA))

        # 2) recommendations (same protocol as in 02_ease_baseline)
        recs = model.recommend(
            train_in,
            train_in,
            top_k=TOP_K,
        )

        # 3) evaluation (reuse central evaluator)
        metrics = evaluate_model(
            recs,
            train_in,
            train_out,
            publisher_mapper=None,     # skip publisher gini for speed
            item_similarity=None,      # skip similarity-based metrics
            item_distance=None,        # skip novelty here (can add if needed)
            k=TOP_K,
        )

        metrics["lambda"] = lam
        results.append(metrics)

    # aggregate and save
    df = pd.DataFrame(results)

    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "results" / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_csv
    df.to_csv(output_path, index=False)

    return df
