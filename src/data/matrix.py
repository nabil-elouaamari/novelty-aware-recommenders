import numpy as np
from scipy.sparse import csr_matrix

def build_user_item_matrix(df, user2idx, item2idx):
    rows = df["user_id"].map(user2idx)
    cols = df["item_id"].map(item2idx)
    data = np.ones(len(df))

    n_users = len(user2idx)
    n_items = len(item2idx)

    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))