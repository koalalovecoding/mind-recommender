# ============================================================
# 13 - Build Full MIND Sparse Interaction Matrices
#
# Train:
#   history clicks + clicked train impressions
#
# Dev:
#   clicked dev impressions only
#   remove cold-start users/items
#   remove user-item pairs already seen in train
#
# Outputs:
#   data/processed/mindlarge/train_interactions.npz
#   data/processed/mindlarge/dev_interactions.npz
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"

TRAIN_PART_DIR = DATA_DIR / "train_positive_interactions"
DEV_PART_DIR = DATA_DIR / "dev_clicked_impressions"

# Read ten Parquet parts at a time.
PARTS_PER_BATCH = 10


# ------------------------------------------------------------
# Load train-based mappings
# ------------------------------------------------------------

with open(
    DATA_DIR / "user_idx_map.json",
    "r",
    encoding="utf-8",
) as file:
    user_idx_map = json.load(file)

with open(
    DATA_DIR / "item_idx_map.json",
    "r",
    encoding="utf-8",
) as file:
    item_idx_map = json.load(file)

num_users = len(user_idx_map)
num_items = len(item_idx_map)
matrix_shape = (num_users, num_items)


# ------------------------------------------------------------
# Convert integer coordinates into a binary CSR matrix
# ------------------------------------------------------------

def build_binary_matrix(rows, cols):
    """
    Build a sparse matrix and convert duplicate coordinates to 1.
    """

    matrix = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (rows, cols),
        ),
        shape=matrix_shape,
    )

    # Repeated user-item pairs may be summed by CSR construction.
    # The final interaction matrix must remain binary.
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    return matrix


# ------------------------------------------------------------
# Build train matrix from 155 Parquet parts
# ------------------------------------------------------------

def build_train_matrix():

    part_files = sorted(
        TRAIN_PART_DIR.glob("part-*.parquet")
    )

    if not part_files:
        raise FileNotFoundError(
            f"No train Parquet files found in {TRAIN_PART_DIR}"
        )

    train_matrix = csr_matrix(
        matrix_shape,
        dtype=np.float32,
    )

    raw_rows = 0

    for start in range(
        0,
        len(part_files),
        PARTS_PER_BATCH,
    ):

        batch_files = part_files[
            start:start + PARTS_PER_BATCH
        ]

        batch_rows = []
        batch_cols = []

        for part_path in batch_files:

            part_df = pd.read_parquet(
                part_path,
                columns=["user_id", "item_id"],
            )

            rows = part_df["user_id"].map(
                user_idx_map
            )

            cols = part_df["item_id"].map(
                item_idx_map
            )

            if rows.isna().any():
                raise ValueError(
                    f"Unknown train user in {part_path.name}"
                )

            if cols.isna().any():
                raise ValueError(
                    f"Unknown train item in {part_path.name}"
                )

            batch_rows.append(
                rows.to_numpy(dtype=np.int32)
            )

            batch_cols.append(
                cols.to_numpy(dtype=np.int32)
            )

            raw_rows += len(part_df)

        batch_matrix = build_binary_matrix(
            rows=np.concatenate(batch_rows),
            cols=np.concatenate(batch_cols),
        )

        # Element-wise maximum keeps repeated pairs equal to 1.
        train_matrix = train_matrix.maximum(
            batch_matrix
        ).tocsr()

        processed = min(
            start + PARTS_PER_BATCH,
            len(part_files),
        )

        print(
            f"Train parts processed: "
            f"{processed}/{len(part_files)}",
            flush=True,
        )

    train_matrix.sort_indices()

    return train_matrix, raw_rows, len(part_files)


# ------------------------------------------------------------
# Build warm-start dev matrix
# ------------------------------------------------------------

def build_dev_matrix(train_matrix):

    part_files = sorted(
        DEV_PART_DIR.glob("part-*.parquet")
    )

    if not part_files:
        raise FileNotFoundError(
            f"No dev Parquet files found in {DEV_PART_DIR}"
        )

    dev_rows = []
    dev_cols = []

    raw_rows = 0
    warm_rows = 0

    cold_users = set()
    cold_items = set()

    for part_path in part_files:

        part_df = pd.read_parquet(
            part_path,
            columns=["user_id", "item_id"],
        )

        raw_rows += len(part_df)

        rows = part_df["user_id"].map(
            user_idx_map
        )

        cols = part_df["item_id"].map(
            item_idx_map
        )

        cold_user_mask = rows.isna()
        cold_item_mask = cols.isna()

        cold_users.update(
            part_df.loc[
                cold_user_mask,
                "user_id",
            ].unique()
        )

        cold_items.update(
            part_df.loc[
                cold_item_mask,
                "item_id",
            ].unique()
        )

        warm_mask = (
            ~cold_user_mask
            & ~cold_item_mask
        )

        warm_rows += int(warm_mask.sum())

        dev_rows.append(
            rows[warm_mask].to_numpy(
                dtype=np.int32
            )
        )

        dev_cols.append(
            cols[warm_mask].to_numpy(
                dtype=np.int32
            )
        )

    dev_matrix = build_binary_matrix(
        rows=np.concatenate(dev_rows),
        cols=np.concatenate(dev_cols),
    )

    # Remove dev positives already present in train.
    train_seen = dev_matrix.multiply(
        train_matrix
    )

    train_seen_pairs = train_seen.nnz

    dev_matrix = (
        dev_matrix - train_seen
    ).tocsr()

    dev_matrix.eliminate_zeros()
    dev_matrix.data[:] = 1.0
    dev_matrix.sort_indices()

    return (
        dev_matrix,
        raw_rows,
        warm_rows,
        len(cold_users),
        len(cold_items),
        train_seen_pairs,
        len(part_files),
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    start_time = time.perf_counter()

    print(
        "Full MIND matrix shape:",
        matrix_shape,
    )

    print("\nBuilding train matrix...")

    (
        train_matrix,
        train_raw_rows,
        train_part_count,
    ) = build_train_matrix()

    print("\nBuilding dev matrix...")

    (
        dev_matrix,
        dev_raw_rows,
        dev_warm_rows,
        cold_user_count,
        cold_item_count,
        train_seen_pairs,
        dev_part_count,
    ) = build_dev_matrix(
        train_matrix
    )

    train_path = (
        DATA_DIR / "train_interactions.npz"
    )

    dev_path = (
        DATA_DIR / "dev_interactions.npz"
    )

    save_npz(
        train_path,
        train_matrix,
        compressed=True,
    )

    save_npz(
        dev_path,
        dev_matrix,
        compressed=True,
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    train_density = (
        train_matrix.nnz
        / (num_users * num_items)
    )

    dev_density = (
        dev_matrix.nnz
        / (num_users * num_items)
    )

    print("\nFull MIND sparse matrices complete.")

    print("\nTrain:")
    print("  Parquet parts:", train_part_count)
    print("  Raw positive rows:", f"{train_raw_rows:,}")
    print("  Matrix shape:", train_matrix.shape)
    print("  Unique positive pairs:", f"{train_matrix.nnz:,}")
    print("  Density:", f"{train_density:.10f}")

    print("\nDev:")
    print("  Parquet parts:", dev_part_count)
    print("  Raw clicked rows:", f"{dev_raw_rows:,}")
    print("  Warm-start rows:", f"{dev_warm_rows:,}")
    print("  Cold-start users:", f"{cold_user_count:,}")
    print("  Cold-start items:", f"{cold_item_count:,}")
    print(
        "  Train-seen pairs removed:",
        f"{train_seen_pairs:,}",
    )
    print("  Matrix shape:", dev_matrix.shape)
    print("  Final positive pairs:", f"{dev_matrix.nnz:,}")
    print("  Density:", f"{dev_density:.10f}")

    print(
        "\nProcessing time:",
        f"{elapsed_time:.2f} seconds",
    )

    print("\nSaved:")
    print(train_path)
    print(dev_path)


if __name__ == "__main__":
    main()