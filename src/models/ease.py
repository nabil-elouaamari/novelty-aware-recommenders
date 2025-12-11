import numpy as np
from scipy.sparse import csr_matrix
from src.models.base_model import BaseRecommender
from src.pipelines.recommend import generate_recommendations
import time


class EASE(BaseRecommender):
    def __init__(self, lambda_reg: float = 100.0):
        """
        EASE model for item-item collaborative filtering.
        :param lambda_reg: Regularization hyperparameter.
        """
        self.lambda_reg = float(lambda_reg)
        self.W = None # item item weight matrix, shape (n_items, n_items)

    def fit(self, X: csr_matrix):
        """
        X shape (n_users, n_items), binary implicit feedback matrix
        """

        t0 = time.time()
        print("\n[EASE.fit] starting")
        print(f"  X shape:    {X.shape}")
        print(f"  X nnz:      {X.nnz}")
        print(
            f"  avg items per user: {X.nnz / X.shape[0]:.2f}  "
            f"avg users per item: {X.nnz / X.shape[1]:.2f}"
        )
        print(f"  lambda_reg: {self.lambda_reg}")

        # Gram matrix G = X^T X in float64 for stability
        t1 = time.time()
        G = (X.T @ X).toarray().astype(np.float64)
        print(
            f"  built G (X^T X) with shape {G.shape} in {time.time() - t1:.2f} s"
        )

        # inspect G diagonal before regularization
        diag_G = np.diag(G)
        print(
            f"  G diag before reg - min: {diag_G.min():.2f}, "
            f"max: {diag_G.max():.2f}, mean: {diag_G.mean():.2f}"
        )

        # add regularization on diagonal
        idx = np.arange(G.shape[0])
        G[idx, idx] += self.lambda_reg

        diag_G_reg = np.diag(G)
        print(
            f"  G diag after reg  - min: {diag_G_reg.min():.2f}, "
            f"max: {diag_G_reg.max():.2f}, mean: {diag_G_reg.mean():.2f}"
        )

        # inverse
        t2 = time.time()
        P = np.linalg.inv(G)
        print(f"  inverted G in {time.time() - t2:.2f} s")

        # compute item item weights
        B = P / -np.diag(P)
        np.fill_diagonal(B, 0.0)

        # sanity checks on W
        nonzero_mask = B != 0
        n_nonzero = nonzero_mask.sum()
        print(f"  W nonzeros: {n_nonzero}")
        print(
            f"  W density:  {n_nonzero / (B.shape[0] * B.shape[1]):.6f}"
        )

        diag_W = np.diag(B)
        print(
            f"  W diag min: {diag_W.min():.6f}, "
            f"max: {diag_W.max():.6f}, mean: {diag_W.mean():.6f}"
        )

        # distribution of absolute weights for a quick feel
        abs_vals = np.abs(B[nonzero_mask])
        print(
            "  |W| stats    - min: {:.6f}, p25: {:.6f}, median: {:.6f}, "
            "p75: {:.6f}, max: {:.6f}".format(
                abs_vals.min(),
                np.percentile(abs_vals, 25),
                np.median(abs_vals),
                np.percentile(abs_vals, 75),
                abs_vals.max(),
            )
        )

        self.W = B
        print(f"[EASE.fit] done in {time.time() - t0:.2f} s\n")
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