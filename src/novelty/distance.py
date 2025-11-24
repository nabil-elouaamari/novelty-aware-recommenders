import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity

def build_genre_feature_matrix(games_df):
    """
    Build multi-hot genre-matrix-shaped (num_items, num_genres).
    Items must be sorted by item_id (important!).
    """

    # Ensure sorted by item_id
    games_df = games_df.sort_values("item_id").reset_index(drop=True)

    # collect all genres
    all_genres = set()
    for row in games_df["genres"].fillna("[]"):
        try:
            all_genres |= set(ast.literal_eval(row))
        except:
            pass

    genre_list = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    # build matrix
    F = np.zeros((len(games_df), len(genre_list)), dtype=np.float32)

    for idx, row in games_df.iterrows():
        try:
            genres = ast.literal_eval(row["genres"]) if pd.notnull(row["genres"]) else []
        except:
            genres = []

        for g in genres:
            if g in genre_to_idx:
                F[idx, genre_to_idx[g]] = 1.0

    return F


def build_genre_similarity_matrix(games_df):
    """
    cosine_similarity ∈ [0,1]
    """
    F = build_genre_feature_matrix(games_df)
    S = cosine_similarity(F)
    return S


def build_genre_distance_matrix(games_df):
    """
    distance = 1 - cosine_similarity ∈ [0,1]
    """
    S = build_genre_similarity_matrix(games_df)
    D = 1 - S
    return D
