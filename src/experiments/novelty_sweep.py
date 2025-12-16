import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.data.loader import load_interactions, load_games
from src.evaluation.splitter import split_train_in_out
from src.evaluation.evaluator import evaluate_model
from src.models.ease import EASE
from src.novelty.rerank import rerank_with_novelty
from src.config import LAMBDA_REG, TOP_K, N_EVAL_USERS, SEED, POP_ALPHA

# --------------------------------------------------
# Helpers to load precomputed genre matrices - If you do not have them, run `notebooks/04-feature-engineering.ipynb` first
# --------------------------------------------------
def load_genre_similarity(path="../data/processed/genre_similarity.npy"):
    return np.load(path)

def load_genre_distance(path="../data/processed/genre_distance.npy"):
    return np.load(path)

# --------------------------------------------------
# Main sweep
# --------------------------------------------------
def run_novelty_sweep(
    ease_lambda_reg: float = LAMBDA_REG,
    novelty_lambdas = None,
    top_k_eval: int = TOP_K,
    top_k_base: int = 100,
    n_eval_users: int = N_EVAL_USERS,
    seed: int = SEED,
    output_csv: str = "novelty_sweep_genre.csv",
):
    """
    - Train EASE with fixed lambda_reg on fold-in interactions.
    - Generate base candidate list per user (top_k_base).
    - For each novelty weight lambda_nov, apply re-ranking and evaluate.
    """

    if novelty_lambdas is None:
        # include 0 (no novelty) plus a range
        novelty_lambdas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    rng = np.random.default_rng(seed)

    # 1) Load data
    train = load_interactions(train=True)
    games = load_games()
    publisher_mapper = games.set_index("item_id")["publisher"] # maps item_id -> publisher (needed for publisher Gini)

    # 2) Codabench-like split: 1 holdout per user
    train_in_full, train_out_full = split_train_in_out(train, seed=seed)

    # users that actually have a holdout
    all_users = train_out_full["user_id"].unique()
    n_eval = min(n_eval_users, len(all_users))
    sample_users = rng.choice(all_users, size=n_eval, replace=False)

    train_in = train_in_full[train_in_full["user_id"].isin(sample_users)].reset_index(drop=True)
    train_out = train_out_full[train_out_full["user_id"].isin(sample_users)].reset_index(drop=True)

    print(f"[INFO] Novelty sweep on {n_eval} users "
          f"(train_in rows = {len(train_in)}, train_out rows = {len(train_out)})")

    # 3) Load precomputed genre matrices
    item_similarity = load_genre_similarity()
    item_distance = load_genre_distance()

    # 4) Train EASE once and get base recommendations
    model = EASE(lambda_reg=ease_lambda_reg, alpha_pop=POP_ALPHA)
    recs_base = model.recommend(train_in, train_in, top_k=top_k_base)

    # 5) Evaluate plain EASE (no novelty) as lambda_nov = 0 baseline
    print("[INFO] Evaluating baseline (lambda_nov = 0)")
    metrics_base = evaluate_model(
        recs_base,
        train_in,
        train_out,
        publisher_mapper=publisher_mapper,
        item_similarity=item_similarity,
        item_distance=item_distance,
        k=top_k_eval,
    )
    metrics_base["ease_lambda_reg"] = ease_lambda_reg
    metrics_base["novelty_lambda"] = 0.0

    results = [metrics_base]

    # 6) Sweep novelty weights
    for lam in tqdm(novelty_lambdas, desc="Sweeping novelty λ"):
        if lam == 0.0:
            # already done
            continue

        # re-rank
        recs_novel = rerank_with_novelty(
            recs_base,
            train_in,
            item_distance,
            lambda_val=lam,
        )

        # very important: sort by user, then score descending
        recs_novel = recs_novel.sort_values(
            ["user_id", "score"], ascending=[True, False]
        ).reset_index(drop=True)

        metrics = evaluate_model(
            recs_novel,
            train_in,
            train_out,
            publisher_mapper=publisher_mapper,
            item_similarity=item_similarity,
            item_distance=item_distance,
            k=top_k_eval,
        )

        metrics["ease_lambda_reg"] = ease_lambda_reg
        metrics["novelty_lambda"] = lam

        results.append(metrics)

    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "results" / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    output_path = output_dir / output_csv
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved novelty sweep results to: {output_path}")

    return df