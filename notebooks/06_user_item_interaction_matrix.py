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

print("Step 8 complete.")