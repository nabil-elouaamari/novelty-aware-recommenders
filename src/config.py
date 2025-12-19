"""
Central configuration for experiments and pipelines.

These constants are imported across the project so that:
- experiments remain reproducible
- hyperparameters stay in one place
"""

# EASE model hyperparameters
LAMBDA_REG = 300        # regularization strength for EASE (tuned in 03_ease_tuning.ipynb)
POP_ALPHA = 0.18        # popularity blending weight, 0 means no popularity boost

# Recommendation settings
TOP_K = 20              # number of items to recommend per user

# Offline evaluation settings
N_EVAL_USERS = 4000     # maximum number of users to sample for offline sweeps
SEED = 42               # random seed for all numpy-based sampling

# Interaction filtering
MIN_PLAYTIME = 0        # minimum minutes of playtime to count as an interaction