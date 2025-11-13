def filter_interactions(df, min_user_interactions=5, min_item_interactions=5):
    """
    Remove users/items with very few interactions.
    """
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()

    df = df[df["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
    df = df[df["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)]

    return df


def encode_ids(df, user_col="user_id", item_col="item_id"):
    """
    Convert IDs to 0...N-1 indices for matrix operations.
    """
    df[user_col] = df[user_col].astype("category").cat.codes
    df[item_col] = df[item_col].astype("category").cat.codes
    return df