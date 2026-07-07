# ============================================================
# Phase 2 Part B: ID Mapping and Sparse Matrix Construction (Steps: 7,8,9)
#
# This script converts the cleaned MIND-small interaction tables
# from Phase 2 Part A into model-ready sparse user-item matrices.
#
# Main steps:
# 1. Load cleaned train/dev interaction tables with news metadata.
# 2. Create integer ID mappings for users and news items.
# 3. Save user/item mapping files as JSON.
# 4. Build train/dev sparse interaction matrices using positive clicks.
# 5. Save the sparse matrices for downstream recommendation models.
#
# Important:
# The sparse matrix represents positive implicit feedback only.
# Rows with click = 0 are exposed-but-not-clicked impressions and
# are not inserted as strong negative feedback in this matrix.
#
# Inputs:
#   ../data/processed/train_with_news.parquet
#   ../data/processed/dev_with_news.parquet
#
# Outputs:
#   ../data/processed/user_id_map.json
#   ../data/processed/item_id_map.json
#   ../data/processed/user_idx_map.json
#   ../data/processed/item_idx_map.json
#   ../data/processed/train_interactions.npz
#   ../data/processed/dev_interactions.npz
# ============================================================



# Step 7: create ID mappings from train only

import json
import pandas as pd
from pathlib import Path
DATA_DIR = Path("../data/processed")
train_path = DATA_DIR / "train_with_news.parquet"
dev_path = DATA_DIR / "dev_with_news.parquet"
train_df = pd.read_parquet(train_path)
dev_df = pd.read_parquet(dev_path)


user_ids = sorted(train_df["user_id"].unique())
item_ids = sorted(train_df["item_id"].unique())

# raw user_id -> integer user index
# Example: "U12345" -> 0
user_idx_map = {user_id: idx for idx, user_id in enumerate(user_ids)}

# raw item_id -> integer item index
# Example: "N67890" -> 15
item_idx_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

# integer user index -> raw user_id
# Example: "0" -> "U12345"
idx_user_map = {str(idx): user_id for user_id, idx in user_idx_map.items()}

# integer item index -> raw item_id
# Example: "15" -> "N67890"
idx_item_map = {str(idx): item_id for item_id, idx in item_idx_map.items()}


# Save mappings
with open(DATA_DIR/"user_idx_map.json","w") as f:
    json.dump(user_idx_map,f)

with open(DATA_DIR/"item_idx_map.json","w") as f:
    json.dump(item_idx_map,f)

with open(DATA_DIR/"idx_user_map.json", "w") as f:
    json.dump(idx_user_map,f)

with open(DATA_DIR/"idx_item_map.json","w") as f:
    json.dump(idx_item_map,f)

#Basic Output
print("Number of train users:", len(user_idx_map))
print("Number of train items:", len(item_idx_map))


# Dev cold-start check
# cold-start means a user/item appears in dev but does not appear in train.
# Basic collaborative filtering models cannot handle these unseen users/items directly.
dev_cold_users = set(dev_df['user_id'].unique())-set(user_idx_map.keys())
dev_cold_items = set(dev_df["item_id"].unique())-set(item_idx_map.keys())

print("dev cold-start users:", len(dev_cold_users))
print("dev cold-start items:", len(dev_cold_items))

