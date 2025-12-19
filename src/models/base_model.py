"""
Abstract base class for recommendation models.

All recommenders in this project should follow the same minimal API:
  - Fit(X): train on a user-item matrix
  - Predict_user(user_vector): score items for a single user
"""

import abc
import numpy as np
from scipy.sparse import csr_matrix


class BaseRecommender(abc.ABC):
    """
    Minimal interface that all recommenders in this project implement.
    """

    @abc.abstractmethod
    def fit(self, X: csr_matrix):
        """
        Train the model on a user item interaction matrix.

        Parameters
        ----------
        X : csr_matrix
            Sparse matrix of shape (n_users, n_items) with binary
            implicit feedback.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_user(self, user_vector: np.ndarray) -> np.ndarray:
        """
        Predict scores for a single user.

        Parameters
        ----------
        user_vector : np.ndarray
            Binary vector of shape (n_items,) with 1 for items the user
            has interacted with and 0 otherwise.

        Returns
        -------
        np.ndarray
            Score vector of shape (n_items,) where higher values mean
            higher relevance.
        """
        raise NotImplementedError
