import numpy as np
from src.models.ease import EASE


def test_ease_fit(sample_matrix):
    model = EASE(lambda_reg=1.0)
    model.fit(sample_matrix)

    assert model.W is not None
    # Weight matrix should be square (n_items x n_items)
    assert model.W.shape == (3, 3)
    # Diagonal should be zero
    assert np.all(np.diag(model.W) == 0)


def test_ease_predict_user(sample_matrix):
    model = EASE(lambda_reg=1.0)
    model.fit(sample_matrix)

    user_vec = np.array([1, 0, 0])
    scores = model.predict_user(user_vec)

    assert scores.shape == (3,)
    # Basic check: scores should not be all zero given the connections
    assert np.sum(np.abs(scores)) > 0
