import numpy as np
from scipy.sparse import csr_matrix
from src.models.base_model import BaseRecommender
from src.pipelines.recommend import generate_recommendations


class EASE(BaseRecommender):
    def __init__(self, lambda_reg: float = 100.0):
        """
        EASE model for item-item collaborative filtering.
        :param lambda_reg: Regularization hyperparameter.
        """
        self.lambda_reg = float(lambda_reg)
        self.W = None # item-item weight matrix, shape (n_items, n_items)

    def fit(self, X: csr_matrix):
        """
        X shape (n_users, n_items), binary implicit feedback matrix
        """

        # Gram matrix G = X^T X in float64 for stability
        G = (X.T @ X).toarray().astype(np.float64)

        # add regularization on diagonal
        idx = np.arange(G.shape[0])
        G[idx, idx] += self.lambda_reg

        # inverse
        P = np.linalg.inv(G)

        # compute item-item weights
        B = P / -np.diag(P)
        np.fill_diagonal(B, 0.0)

        self.W = B
        return self

    def predict_user(self, user_vector: np.ndarray) -> np.ndarray:
        """
        Predict scores for a single user.
        :param user_vector: binary interaction vector for user u. (1 for interacted items, 0 otherwise)
        :return: Predicted scores vector for user u.
        """
        if self.W is None:
            raise RuntimeError("Call fit(X) before predict_user")
        return user_vector @ self.W

    def recommend(self, train_df, test_in_df, top_k: int = 20):
        return generate_recommendations(self, train_df, test_in_df, top_k)


if __name__ == "__main__":
    ### example usage
    # Sample user-item interaction matrix
    _X = csr_matrix([[1, 0, 1],  # User 1 interacted with items 0 and 2
                    [0, 1, 0],  # User 2 interacted with item 1
                    [1, 1, 0]]) # User 3 interacted with items 0 and 1
    model = EASE()
    model.fit(_X)
    _user_vector = np.array([1, 0, 0])  # Example: User interacted with item 0
    scores = model.predict_user(_user_vector)

    print("Predicted scores:", scores)
    # output (lambda_reg:5.0): Predicted scores: [0.         0.14634146 0.14583333]