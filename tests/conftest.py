import pytest
import pandas as pd
from scipy.sparse import csr_matrix

@pytest.fixture
def sample_interactions():
    # Simple interaction data: 3 users, 4 items
    data = {
        "user_id": [1, 1, 2, 3, 3, 3],
        "item_id": [10, 11, 11, 10, 12, 13],
        "score": [1, 1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_matrix():
    # 3 users x 3 items
    return csr_matrix([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1]
    ])