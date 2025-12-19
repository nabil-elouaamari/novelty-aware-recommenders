"""
EASE model (Embarrassingly Simple Autoencoder).

This is an item item collaborative filtering model with a closed form
solution. It learns an item item weight matrix W and scores a user by
multiplying their interaction vector with W.

On top of the pure EASE scores, this implementation supports blending
with simple item popularity prior controlled by alpha_pop.
"""

import numpy as np
from scipy.sparse import csr_matrix

from src.models.base_model import BaseRecommender
from src.pipelines.recommend import generate_recommendations


class EASE(BaseRecommender):
    """
    EASE item item collaborative filtering model.

    The core idea:
      - Build Gram matrix G = X^T X from a user item matrix X
      - Add lambda_reg on the diagonal of G for regularization
      - Invert G to get P
      - Derive item item weights W from P and zero the diagonal

    Optionally, scores can be blended with an item popularity signal.

    Parameters
    ----------
    lambda_reg : float, default 100.0
        Regularization strength added on the diagonal of X^T X.
        Higher values shrink item item weights and usually reduce
        overfitting.
    alpha_pop : float, default 0.15
        Blending weight for item popularity in the final score.
        The final score is:
            (1 - alpha_pop) * ease_score + alpha_pop * pop_score
        Set to 0.0 to disable popularity blending.
    """

    def __init__(self, lambda_reg: float = 100.0, alpha_pop: float = 0.15):
        self.lambda_reg = float(lambda_reg)
        self.W = None  # item item weight matrix, shape (n_items, n_items)

        # popularity blending
        self.alpha_pop = float(alpha_pop)
        self.popularity = None  # vector of length n_items, normalized to [0, 1]

    def fit(self, X: csr_matrix):
        """
        Fit EASE on a user item matrix and compute popularity.

        Parameters
        ----------
        X : csr_matrix
            Sparse binary matrix of shape (n_users, n_items).

        Returns
        -------
        self : EASE
            The fitted model.
        """
        # Gram matrix G = X^T X in float64 for numerical stability
        G = (X.T @ X).toarray().astype(np.float64)

        # add regularization on the diagonal
        idx = np.arange(G.shape[0])
        G[idx, idx] += self.lambda_reg

        # invert G
        P = np.linalg.inv(G)

        # compute item item weights
        B = P / -np.diag(P)
        np.fill_diagonal(B, 0.0)

        self.W = B

        # compute item popularity from training matrix
        # use column sums (number of users interacted), apply log1p and normalize
        item_pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        if item_pop.size > 0:
            item_pop = np.log1p(item_pop)  # compress heavy tail
            max_val = item_pop.max()
            if max_val > 0:
                item_pop = item_pop / max_val
        self.popularity = item_pop

        return self

    def predict_user(self, user_vector: np.ndarray) -> np.ndarray:
        """
        Predict scores for a single user.

        Parameters
        ----------
        user_vector : np.ndarray
            Binary interaction vector of shape (n_items,).
            1 means the user has interacted with the item
            0 means no interaction.

        Returns
        -------
        np.ndarray
            Score vector of shape (n_items,).
        """
        if self.W is None:
            raise RuntimeError("Call fit(X) before predict_user")

        # pure EASE scores
        ease_scores = user_vector @ self.W

        # optional popularity blending
        if (
            self.alpha_pop
            and (self.popularity is not None)
            and len(self.popularity) == ease_scores.shape[0]
        ):
            scores = (1.0 - self.alpha_pop) * ease_scores + self.alpha_pop * self.popularity
        else:
            scores = ease_scores

        return scores

    def recommend(
        self,
        train_df,
        test_in_df,
        top_k: int = 20,
        min_playtime: int | None = None,
    ):
        """
        Convenience wrapper around generate_recommendations so the
        model can be used with a single call.

        Parameters
        ----------
        train_df : pd.DataFrame
            Interactions used to fit the model.
        test_in_df : pd.DataFrame
            Fold in interactions that define which users and histories
            we score.
        top_k : int, default 20
            Number of recommendations per user.
        min_playtime : int or None, default None
            Optional filter that ignores very short playtime
            interactions when building histories and the training
            matrix. If None, no playtime filter is applied.

        Returns
        -------
        pd.DataFrame
            Recommendation list with columns
            ['user_id', 'item_id', 'score', 'rank'].
        """
        return generate_recommendations(
            self,
            train_df,
            test_in_df,
            top_k,
            min_playtime=min_playtime,
        )


if __name__ == "__main__":
    # minimal example usage
    from scipy.sparse import csr_matrix

    # Sample user item interaction matrix
    _X = csr_matrix(
        [
            [1, 0, 1],  # User 0 interacted with items 0 and 2
            [0, 1, 0],  # User 1 interacted with item 1
            [1, 1, 0],  # User 2 interacted with items 0 and 1
        ]
    )
    model = EASE()
    model.fit(_X)

    _user_vector = np.array([1, 0, 0])  # Example: user interacted with item 0
    scores = model.predict_user(_user_vector)
    print("Predicted scores:", scores)