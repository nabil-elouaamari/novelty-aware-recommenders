import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

def load_interactions(train=True):
    filename = "train_interactions.csv" if train else "test_interactions_in.csv"
    return pd.read_csv(DATA_DIR / filename)

def load_games():
    return pd.read_csv(DATA_DIR / "games.csv")

def load_extended_games():
    return pd.read_csv(DATA_DIR / "extended_games.csv")

def load_item_reviews():
    return pd.read_csv(DATA_DIR / "item_reviews.csv")

def load_user_reviews():
    return pd.read_csv(DATA_DIR / "user_reviews.csv")

def load_bundles():
    return pd.read_csv(DATA_DIR / "bundles.csv")
