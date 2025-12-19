import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity


def build_genre_feature_matrix(games_df: pd.DataFrame) -> np.ndarray:
    """
    Build a multi-hot genre feature matrix of shape (n_items, n_genres).

    Parameters
    ----------
    games_df : pd.DataFrame
        Must contain at least:
          - "item_id": integer item id used in interactions
          - "genres": stringifies a Python list, for example, "['Action', 'RPG']"

    Returns
    -------
    np.ndarray
        Matrix F with shape (n_items, n_genres) where:
          - Rows are sorted by item_id
          - F[i, g] = 1.0 if item i has genre g, otherwise 0.0

    Notes
    -----
    The sorting by item_id is important. All later distance and similarity
    matrices assume that item_id i corresponds to row i in F.
    """
    # ensure sorted by item_id so that row index == item_id
    games_df = games_df.sort_values("item_id").reset_index(drop=True)

    # collect all unique genres
    all_genres: set[str] = set()
    for row in games_df["genres"].fillna("[]"):
        try:
            all_genres |= set(ast.literal_eval(row))
        except Exception:
            # if parsing fails, just skip this row
            pass

    genre_list = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    # build multi hot matrix
    F = np.zeros((len(games_df), len(genre_list)), dtype=np.float32)

    for idx, row in games_df.iterrows():
        try:
            genres = ast.literal_eval(row["genres"]) if pd.notnull(row["genres"]) else []
        except Exception:
            genres = []

        for g in genres:
            if g in genre_to_idx:
                F[idx, genre_to_idx[g]] = 1.0

    return F


def build_genre_similarity_matrix(games_df: pd.DataFrame) -> np.ndarray:
    """
    Compute a cosine similarity matrix between items in genre space.

    Parameters
    ----------
    games_df : pd.DataFrame
        See build_genre_feature_matrix for required columns.

    Returns
    -------
    np.ndarray
        Similarity matrix S with shape (n_items, n_items) where
        S[i, j] is the cosine similarity between the genre vectors of items i and j.
        Values are in [0, 1].
    """
    F = build_genre_feature_matrix(games_df)
    S = cosine_similarity(F)
    return S


def build_genre_distance_matrix(games_df: pd.DataFrame) -> np.ndarray:
    """
    Compute a cosine distance matrix between items in genre space.

    Parameters
    ----------
    games_df : pd.DataFrame
        See build_genre_feature_matrix for required columns.

    Returns
    -------
    np.ndarray
        Distance matrix D with shape (n_items, n_items) where
        D[i, j] = 1 - cosine_similarity(F[i], F[j]) and values are in [0, 1].

    Notes
    -----
    This matrix is used to define item to history novelty, where higher
    distances correspond to more novel games for a user.
    """
    S = build_genre_similarity_matrix(games_df)
    D = 1 - S
    return D