import pandas as pd
import numpy as np

from src.pipelines.recommend import generate_recommendations


class DummyModel:
    """
    Tiny stand in model for testing generate_recommendations.

    - fit(X) just remembers the number of items
    - predict_user(user_vector) returns deterministic scores 0..n_items-1 so a higher item index means a higher score.
    """
    def __init__(self):
        self.W = None
        self.n_items = None

    def fit(self, X):
        self.n_items = X.shape[1]
        # Just mark the model as "fitted" so generate_recommendations
        # does not call fit again later.
        self.W = True

    def predict_user(self, user_vector: np.ndarray) -> np.ndarray:
        assert len(user_vector) == self.n_items
        # simple deterministic scores: score = item index
        return np.arange(self.n_items, dtype=np.float32)


def test_generate_recommendations_basic():
    """
    - One user, items 0 and 2 in history.
    - Item 1 is unseen.
    - With our DummyModel scores = [0,1,2], after masking history (0,2)
      only item 1 should be recommended when top_k=1.
    """
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [0, 2],
        }
    )
    # offline style: fold in equals train here
    test_in_df = train_df.copy()

    model = DummyModel()
    recs = generate_recommendations(model, train_df, test_in_df, top_k=1)

    assert not recs.empty
    assert {"user_id", "item_id", "score", "rank"}.issubset(recs.columns)
    assert recs["user_id"].nunique() == 1

    # user already interacted with items 0 and 2
    # so the single recommendation should be item 1
    rec_item_ids = recs["item_id"].tolist()
    assert rec_item_ids == [1]


def test_generate_recommendations_respects_min_playtime():
    """
    Check that min_playtime filtering is used consistently:
    - user has item 0 with playtime 1 and item 1 with playtime 10
    - with min_playtime=5, only item 1 counts as "seen"
      so item 0 can be recommended again.
    """
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [0, 1],
            "playtime": [1, 10],
        }
    )
    test_in_df = train_df.copy()

    model = DummyModel()
    recs = generate_recommendations(
        model,
        train_df,
        test_in_df,
        top_k=1,
        min_playtime=5,
    )

    assert not recs.empty
    # because item 1 passes the playtime threshold, it is masked,
    # so item 0 should be recommended
    assert recs.iloc[0]["item_id"] == 0
