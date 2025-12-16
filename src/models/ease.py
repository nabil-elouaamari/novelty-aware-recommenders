import numpy as np
from scipy.sparse import csr_matrix
from src.models.base_model import BaseRecommender
from src.pipelines.recommend import generate_recommendations


class EASE(BaseRecommender):
    def __init__(self, lambda_reg: float = 100.0, alpha_pop: float = 0.15):
        """
        EASE model for item-item collaborative filtering.

        :param lambda_reg: Regularization hyperparameter.
        :param alpha_pop: blending weight for item popularity (0 = no pop boost).
                          Final score = (1-alpha) * ease_score + alpha * pop_score
        """
        self.lambda_reg = float(lambda_reg)
        self.W = None  # item-item weight matrix, shape (n_items, n_items)

        # popularity blending
        self.alpha_pop = float(alpha_pop)
        self.popularity = None  # vector of length n_items, normalized to [0,1]

    def fit(self, X: csr_matrix):
        """
        X shape (n_users, n_items), binary implicit feedback matrix

        Also computes item popularity used for score blending.
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

        # compute item popularity from training matrix
        # use column sums (number of users interacted), apply log1p and normalize to [0,1]
        item_pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        if item_pop.size > 0:
            item_pop = np.log1p(item_pop)  # compress heavy-tail
            max_val = item_pop.max()
            if max_val > 0:
                item_pop = item_pop / max_val
            else:
                item_pop = item_pop
        self.popularity = item_pop

        return self

    def predict_user(self, user_vector: np.ndarray) -> np.ndarray:
        """
        Predict scores for a single user.
        :param user_vector: binary interaction vector for user u. (1 for interacted items, 0 otherwise)
        :return: Predicted scores vector for user u.
        """
        if self.W is None:
            raise RuntimeError("Call fit(X) before predict_user")
        ease_scores = user_vector @ self.W

        # if popularity blending is enabled and vector available, blend scores
        if self.alpha_pop and (self.popularity is not None) and len(self.popularity) == ease_scores.shape[0]:
            scores = (1.0 - self.alpha_pop) * ease_scores + self.alpha_pop * self.popularity
        else:
            scores = ease_scores

        return scores

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