import pandas as pd
from src.data.preprocess import filter_interactions, encode_ids


def test_filter_interactions():
    # Create data where user '2' has only 1 interaction
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 3, 3, 3],
        "item_id": [10, 11, 12, 10, 10, 11, 12]
    })

    # Filter min 2 interactions
    filtered = filter_interactions(df, min_user_interactions=2, min_item_interactions=1)

    assert 2 not in filtered["user_id"].values
    assert 1 in filtered["user_id"].values
    assert len(filtered) == 6


def test_encode_ids():
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "item_id": ["i1", "i2"]
    })
    encoded = encode_ids(df)

    # Check if columns are now integers (codes)
    assert pd.api.types.is_integer_dtype(encoded["user_id"])
    assert pd.api.types.is_integer_dtype(encoded["item_id"])
    assert encoded["user_id"].min() == 0
