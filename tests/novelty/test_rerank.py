import numpy as np
import pandas as pd

from src.novelty.rerank import rerank_with_novelty


def test_rerank_with_novelty_increases_scores_for_distant_items():
    """
    Check that items further from the user's history get a larger score boost.
    """
    # Two candidate items for the same user
    recs_df = pd.DataFrame(
        {
            "user_id": [0, 0],
            "item_id": [0, 1],
            "score": [1.0, 1.0],
        }
    )

    # User 0 has seen items 2 and 3
    test_in = pd.DataFrame(
        {
            "user_id": [0, 0],
            "item_id": [2, 3],
        }
    )

    # 4 items total: 0,1,2,3
    item_distance = np.zeros((4, 4), dtype=np.float32)

    # Make item 0 far from history (distance 1), item 1 close (distance 0)
    item_distance[0, 2] = 1.0
    item_distance[0, 3] = 1.0
    item_distance[2, 0] = 1.0
    item_distance[3, 0] = 1.0
    # everything else stays at 0

    lambda_val = 0.5
    reranked = rerank_with_novelty(
        recs_df,
        test_in,
        item_distance,
        lambda_val=lambda_val,
    )

    # Extract scores
    s_item0 = float(
        reranked.loc[reranked["item_id"] == 0, "score"].iloc[0]
    )
    s_item1 = float(
        reranked.loc[reranked["item_id"] == 1, "score"].iloc[0]
    )

    # item 0 should have received a positive boost, item 1 no boost
    assert s_item0 == 1.0 + lambda_val * 1.0
    assert s_item1 == 1.0
    assert s_item0 > s_item1
