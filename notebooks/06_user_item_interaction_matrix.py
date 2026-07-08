# ============================================================
# Phase 2 Part B - Step 8: Sparse Matrix Construction
#
# This script builds train/dev sparse user-item interaction matrices
# using the ID mappings created in Step 7.
#
# The matrix contains positive implicit feedback only:
# R[user_idx, item_idx] = 1 if the user clicked the item.
# Rows with click = 0 are not inserted as strong negative feedback.
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


DATA_DIR = Path("../data/processed")

# Load cleaned interaction tables
train_df = pd.read_parquet(DATA_DIR / "train_with_news.parquet")
dev_df = pd.read_parquet(DATA_DIR / "dev_with_news.parquet")

# Load mappings from Step 7
with open(DATA_DIR / "user_idx_map.json", "r") as f:
    user_idx_map = json.load(f)

with open(DATA_DIR / "item_idx_map.json", "r") as f:
    item_idx_map = json.load(f)

num_users = len(user_idx_map)
num_items = len(item_idx_map)


# -----------------------------
# Train sparse matrix
# -----------------------------
train_pos = train_df[train_df["click"] == 1][["user_id", "item_id"]].drop_duplicates()

train_rows = train_pos["user_id"].map(user_idx_map)
train_cols = train_pos["item_id"].map(item_idx_map)

train_matrix = csr_matrix(
    (np.ones(len(train_pos), dtype=np.float32), (train_rows, train_cols)),
    shape=(num_users, num_items),
)


# -----------------------------
# Dev sparse matrix
# Keep only users/items seen in train.
# -----------------------------

# Dev matrix is used as validation ground truth, so use clicked dev impressions only.
# Do not include dev history clicks, because history represents past clicks, not the target to predict
dev_pos = dev_df[
    (dev_df["click"] == 1) &
    (dev_df["source"] == "impression")
][["user_id", "item_id"]].drop_duplicates()

dev_pos = dev_pos[
    dev_pos["user_id"].isin(user_idx_map)
    & dev_pos["item_id"].isin(item_idx_map)
]

# Remove dev positive pairs already seen in train.
# Validation should test future interactions not used as training positives.
train_pair_df = train_pos[["user_id", "item_id"]].drop_duplicates()
dev_pos = dev_pos.merge(
    train_pair_df.assign(seen_in_train=1),
    on=["user_id", "item_id"],
    how="left",
)

print("dev positive pairs already seen in train:", dev_pos["seen_in_train"].notna().sum())

dev_pos = dev_pos[dev_pos["seen_in_train"].isna()]
dev_pos = dev_pos.drop(columns=["seen_in_train"])

dev_rows = dev_pos["user_id"].map(user_idx_map)
dev_cols = dev_pos["item_id"].map(item_idx_map)

dev_matrix = csr_matrix(
    (np.ones(len(dev_pos), dtype=np.float32), (dev_rows, dev_cols)),
    shape=(num_users, num_items),
)


# Save sparse matrices
save_npz(DATA_DIR / "train_interactions.npz", train_matrix)
save_npz(DATA_DIR / "dev_interactions.npz", dev_matrix)

print("train matrix shape:", train_matrix.shape)
print("dev matrix shape:", dev_matrix.shape)
print("train nnz:", train_matrix.nnz)
print("dev nnz:", dev_matrix.nnz)



# -----------------------------
# Extra diagnostic output for dev filtering
# This block only recomputes counts for reporting.
# It does not change the saved train/dev sparse matrices.
# -----------------------------

dev_clicked_impression_pairs = dev_df[
    (dev_df["click"] == 1) &
    (dev_df["source"] == "impression")
][["user_id", "item_id"]].drop_duplicates()

dev_warm_start_pairs = dev_clicked_impression_pairs[
    dev_clicked_impression_pairs["user_id"].isin(user_idx_map)
    & dev_clicked_impression_pairs["item_id"].isin(item_idx_map)
]

train_pairs = set(map(tuple, train_pos[["user_id", "item_id"]].to_numpy()))
dev_pairs = list(map(tuple, dev_warm_start_pairs[["user_id", "item_id"]].to_numpy()))

num_seen_in_train = sum(pair in train_pairs for pair in dev_pairs)
num_final_dev_pairs = len(dev_warm_start_pairs) - num_seen_in_train

print("dev clicked impression pairs before filtering:", len(dev_clicked_impression_pairs))
print("dev warm-start pairs after user/item filtering:", len(dev_warm_start_pairs))
print("dev positive pairs already seen in train:", num_seen_in_train)
print("dev final positive pairs after removing train-seen pairs:", num_final_dev_pairs)

print("Step 8 complete.")