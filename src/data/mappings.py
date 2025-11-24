import pandas as pd
def build_id_mappings(train_df, test_df):
    # USERS
    users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()

    # ITEMS
    items = pd.concat([train_df["item_id"], test_df["item_id"]]).unique()

    user2idx = {u: i for i, u in enumerate(users)}
    idx2user = {i: u for u, i in user2idx.items()}

    item2idx = {i: j for j, i in enumerate(items)}
    idx2item = {j: i for i, j in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item