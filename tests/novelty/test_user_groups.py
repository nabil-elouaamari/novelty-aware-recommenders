import pandas as pd
import numpy as np

from src.novelty.user_groups import (
    assign_groups_by_profile_size,
    assign_groups_by_genre_diversity,
    assign_groups_combined,
)


def test_assign_groups_by_profile_size_thresholds():
    """
    Users with more interactions should get larger novelty weights.
    """
    # user 1: 2 interactions
    # user 2: 10 interactions
    # user 3: 60 interactions
    data = []
    data += [{"user_id": 1, "item_id": i} for i in range(2)]
    data += [{"user_id": 2, "item_id": i} for i in range(2, 12)]
    data += [{"user_id": 3, "item_id": i} for i in range(12, 72)]
    train_in = pd.DataFrame(data)

    user_lambda = assign_groups_by_profile_size(
        train_in,
        small=5,
        medium=20,
    )

    # user 1: small profile
    # user 2: medium profile
    # user 3: large profile
    assert user_lambda[1] == 0.10
    assert user_lambda[2] == 0.20
    assert user_lambda[3] == 0.30


def test_assign_groups_by_genre_diversity_outputs_valid_weights():
    """
    Genre-based grouping should assign one of {0.10, 0.20, 0.30} to each user.
    """
    # Simple game metadata with genres as stringifies lists
    games_df = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "genres": [
                "['Action']",
                "['Action']",
                "['Puzzle']",
                "['Action', 'Puzzle']",
            ],
        }
    )

    # Build some histories
    data = [
        # user 0: pure action
        {"user_id": 0, "item_id": 0},
        {"user_id": 0, "item_id": 1},
        # user 1: mixed action + puzzle
        {"user_id": 1, "item_id": 0},
        {"user_id": 1, "item_id": 2},
        # user 2: only puzzle
        {"user_id": 2, "item_id": 2},
    ]
    train_in = pd.DataFrame(data)

    user_lambda = assign_groups_by_genre_diversity(train_in, games_df)

    # Should have one lambda per user
    assert set(user_lambda.keys()) == {0, 1, 2}

    # All weights must be one of the predefined values
    allowed = {0.10, 0.20, 0.30}
    assert set(user_lambda.values()).issubset(allowed)


def test_assign_groups_combined_is_average():
    """
    Combined strategy should average profile-based and genre-based weights,
    using the same profile weights as assign_groups_combined internally.
    """
    games_df = pd.DataFrame(
        {
            "item_id": [0, 1],
            "genres": [
                "['Action']",
                "['Puzzle']",
            ],
        }
    )
    train_in = pd.DataFrame(
        {
            "user_id": [0, 0, 1],
            "item_id": [0, 1, 0],
        }
    )

    # Use the same profile weights as the combined function (defaults)
    lam_profile_default = assign_groups_by_profile_size(train_in)
    lam_genre = assign_groups_by_genre_diversity(train_in, games_df)
    lam_combined = assign_groups_combined(train_in, games_df)

    for user in lam_combined:
        expected = (lam_profile_default[user] + lam_genre[user]) / 2.0
        assert np.isclose(lam_combined[user], expected)
