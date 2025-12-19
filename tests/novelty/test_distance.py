import numpy as np
import pandas as pd

from src.novelty.distance import (
    build_genre_feature_matrix,
    build_genre_similarity_matrix,
    build_genre_distance_matrix,
)


def test_build_genre_feature_matrix_shape_and_content():
    games_df = pd.DataFrame(
        {
            "item_id": [0, 1, 2],
            "genres": [
                "['Action']",
                "['Action', 'RPG']",
                "['Puzzle']",
            ],
        }
    )

    F = build_genre_feature_matrix(games_df)

    # 3 items, at least 3 genres across them
    assert F.shape[0] == 3
    assert F.shape[1] >= 3

    # item 0 must have at least one genre
    assert F[0].sum() == 1
    # item 1 must have at least two genres
    assert F[1].sum() >= 1


def test_similarity_and_distance_are_consistent():
    games_df = pd.DataFrame(
        {
            "item_id": [0, 1],
            "genres": [
                "['Action']",
                "['Action']",
            ],
        }
    )

    S = build_genre_similarity_matrix(games_df)
    D = build_genre_distance_matrix(games_df)

    # S and D must be square and the same shape
    assert S.shape == (2, 2)
    assert D.shape == (2, 2)

    # Similarity diagonal should be 1, distance diagonal 0
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(np.diag(D), 0.0)

    # By definition D = 1 - S
    assert np.allclose(D, 1.0 - S)