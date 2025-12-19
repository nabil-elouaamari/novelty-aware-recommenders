LAMBDA_REG = 2000    # Regularization parameter for EASE model - chosen based on `notebooks/03_ease_tuning.ipynb`
TOP_K = 20          # How many items to recommend per user
N_EVAL_USERS = 4000 # How many users to evaluate offline
SEED = 42           # Random seed for reproducibility

# popularity blending weight (0 = no popularity boost). Tune this to improve recall / ndcg.
POP_ALPHA = 0.18

# how many minutes count as real interaction?
MIN_PLAYTIME = 0