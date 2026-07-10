# ============================================================
# 12 - Build Full MIND User and Item Mappings
#
# Create train-based integer mappings for all Full MIND users
# and news items.
#
# Inputs:
#   data/raw/MINDlarge_train/behaviors.tsv
#   data/raw/MINDlarge_train/news.tsv
#
# Outputs:
#   data/processed/mindlarge/user_idx_map.json
#   data/processed/mindlarge/item_idx_map.json
#   data/processed/mindlarge/idx_user_map.json
#   data/processed/mindlarge/idx_item_map.json
# ============================================================

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_BEHAVIORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MINDlarge_train"
    / "behaviors.tsv"
)

TRAIN_NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MINDlarge_train"
    / "news.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mindlarge"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()


# ------------------------------------------------------------
# Collect all unique train users
# user_id is the second column in behaviors.tsv.
# ------------------------------------------------------------

user_ids = set()

with open(
    TRAIN_BEHAVIORS_PATH,
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        fields = line.split("\t", 2)

        if len(fields) >= 2:
            user_ids.add(fields[1])

user_ids = sorted(user_ids)


# ------------------------------------------------------------
# Collect all unique train news items
# news_id is the first column in news.tsv.
# ------------------------------------------------------------

item_ids = set()

with open(
    TRAIN_NEWS_PATH,
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        item_id = line.split("\t", 1)[0]

        if item_id:
            item_ids.add(item_id)

item_ids = sorted(item_ids)


# ------------------------------------------------------------
# Create forward and reverse mappings
# ------------------------------------------------------------

# raw user_id -> integer user_idx
user_idx_map = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

# raw item_id -> integer item_idx
item_idx_map = {
    item_id: idx
    for idx, item_id in enumerate(item_ids)
}

# integer user_idx -> raw user_id
idx_user_map = {
    str(idx): user_id
    for user_id, idx in user_idx_map.items()
}

# integer item_idx -> raw item_id
idx_item_map = {
    str(idx): item_id
    for item_id, idx in item_idx_map.items()
}


# ------------------------------------------------------------
# Save mappings
# ------------------------------------------------------------

with open(
    OUTPUT_DIR / "user_idx_map.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(user_idx_map, file)

with open(
    OUTPUT_DIR / "item_idx_map.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(item_idx_map, file)

with open(
    OUTPUT_DIR / "idx_user_map.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(idx_user_map, file)

with open(
    OUTPUT_DIR / "idx_item_map.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(idx_item_map, file)


# ------------------------------------------------------------
# Basic output
# ------------------------------------------------------------

elapsed_time = time.perf_counter() - start_time

print("Number of train users:", len(user_idx_map))
print("Number of train items:", len(item_idx_map))
print(
    "Matrix shape:",
    (len(user_idx_map), len(item_idx_map)),
)
print(
    "Processing time:",
    round(elapsed_time, 2),
    "seconds",
)