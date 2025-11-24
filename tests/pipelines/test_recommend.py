from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from src.pipelines.recommend import generate_recommendations


@patch('src.pipelines.recommend.build_id_mappings')
@patch('src.pipelines.recommend.build_user_item_matrix')
def test_generate_recommendations(mock_matrix, mock_mappings):
    # Setup Mocks
    mock_mappings.return_value = (
        {1: 0}, {0: 1},  # user maps
        {10: 0, 11: 1}, {0: 10, 1: 11}  # item maps
    )

    # Mock matrix 1 user, 2 items
    mock_matrix.return_value = csr_matrix([[1, 0]])

    # Mock Model
    mock_model = MagicMock()
    mock_model.W = np.array([[0, 0.5], [0.5, 0]])  # Simple weight matrix

    train_df = pd.DataFrame({"user_id": [1], "item_id": [10]})
    test_in_df = pd.DataFrame({"user_id": [1], "item_id": [10]})  # Same as train for simplicity

    recs = generate_recommendations(mock_model, train_df, test_in_df, top_k=1)

    assert not recs.empty
    assert "score" in recs.columns
    assert len(recs) == 1
    # Item 10 is in input, so it should be masked (-inf), item 11 should be recommended
    assert recs.iloc[0]["item_id"] == 11