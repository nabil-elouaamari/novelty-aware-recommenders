import abc
import numpy as np
from scipy.sparse import csr_matrix

class BaseRecommender(abc.ABC):
    """
    Abstract base class for recommendation models.
    Ensures all models share the same API.
    """

    @abc.abstractmethod
    def fit(self, X: csr_matrix):
        """Train the model on a user-item matrix."""
        pass

    @abc.abstractmethod
    def predict_user(self, user_vector: np.ndarray) -> np.ndarray:
        """
        Predict scores for a single user.
        user_vector: shape (n_items,)
        Returns: score vector of shape (n_items,)
        """
        pass