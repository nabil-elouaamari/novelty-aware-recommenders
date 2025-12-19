import pytest
from scipy.sparse import csr_matrix

@pytest.fixture
def sample_matrix():
    # 3 users x 3 items
    return csr_matrix([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1]
    ])