# ============================================================
# Phase 2 Part C - Step 14: ALS Recommender
#
# Goal:
# 1. Train an implicit-feedback ALS model on MIND-small.
# 2. Save the trained model and latent factors.
# 3. Generate top-K recommendations for one sample user.
# 4. Filter items already clicked by that user.
#
# Inputs:
#   data/processed/train_interactions.npz
#   data/processed/idx_user_map.json
#   data/processed/idx_item_map.json
#   data/processed/news.parquet
#
# Outputs:
#   data/processed/als_model.npz
#   data/processed/als_user_factors.npy
#   data/processed/als_item_factors.npy
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares


# ------------------------------------------------------------
# Paths and model settings
# ------------------------------------------------------------

# Build paths from the script location, so the script works
# whether it is run from the project root or another directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TOP_K = 10

# ALS hyperparameters
FACTORS = 64 # low rank: K; or the dimension of latent space
REGULARIZATION = 0.1 # to avoid overfitting
ALPHA = 40.0 #c_{ui} = 1 + \alpha R_{ui}
ITERATIONS = 15
RANDOM_STATE = 42


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():

    # --------------------------------------------------------
    # Load the train user-item matrix
    # --------------------------------------------------------

    # Shape:
    #     number of users x number of items
    #
    # A stored value of 1 means that the user clicked the item
    # at least once in the training data.
    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr().astype(np.float32)

    # --------------------------------------------------------
    # Load reverse ID mappings
    # --------------------------------------------------------

    # idx_user_map:
    #     matrix user index -> original MIND user ID
    with open(
        DATA_DIR / "idx_user_map.json",
        "r",
        encoding="utf-8",
    ) as file:
        idx_user_map = json.load(file)

    # idx_item_map:
    #     matrix item index -> original MIND news ID
    with open(
        DATA_DIR / "idx_item_map.json",
        "r",
        encoding="utf-8",
    ) as file:
        idx_item_map = json.load(file)

    # Load news titles for readable recommendation output.
    news_df = pd.read_parquet(
        DATA_DIR / "news.parquet",
        columns=["news_id", "title"],
    ).drop_duplicates(subset=["news_id"])

    title_map = dict(
        zip(
            news_df["news_id"],
            news_df["title"],
        )
    )

    # --------------------------------------------------------
    # Check matrix and mapping consistency
    # --------------------------------------------------------

    if train_matrix.shape[0] != len(idx_user_map):
        raise ValueError(
            "Train matrix rows do not match the user mapping."
        )

    if train_matrix.shape[1] != len(idx_item_map):
        raise ValueError(
            "Train matrix columns do not match the item mapping."
        )

    print("train matrix shape:", train_matrix.shape)
    print("train matrix nnz:", train_matrix.nnz)

    # --------------------------------------------------------
    # Create and train the ALS model
    # --------------------------------------------------------

    # factors:
    #     Dimension of each user and item latent vector.
    #
    # regularization:
    #     Controls the penalty on large factor values.
    #
    # alpha:
    #     Multiplies the confidence of observed positive entries.
    #
    # iterations:
    #     Number of times ALS alternates between updating user
    #     factors and item factors.
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        alpha=ALPHA,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
    )

    print("\nTraining ALS...")

    # The implicit package expects rows to be users and columns
    # to be items, which matches train_interactions.npz.
    model.fit(
        train_matrix,
        show_progress=True,
    )

    print("ALS training complete.")

    # --------------------------------------------------------
    # Check learned latent-factor matrices
    # --------------------------------------------------------

    # user_factors[u] is the latent vector for user u.
    # item_factors[i] is the latent vector for item i.
    print("\nuser factors shape:", model.user_factors.shape)
    print("item factors shape:", model.item_factors.shape)

    expected_user_shape = (
        train_matrix.shape[0],
        FACTORS,
    )

    expected_item_shape = (
        train_matrix.shape[1],
        FACTORS,
    )

    if model.user_factors.shape != expected_user_shape:
        raise ValueError(
            f"Unexpected user factor shape: "
            f"{model.user_factors.shape}"
        )

    if model.item_factors.shape != expected_item_shape:
        raise ValueError(
            f"Unexpected item factor shape: "
            f"{model.item_factors.shape}"
        )

    # --------------------------------------------------------
    # Save the model and factors
    # --------------------------------------------------------

    # Save the complete ALS model.
    model.save(
        str(DATA_DIR / "als_model.npz")
    )

    # Save factors separately because 09 will use them directly
    # as user query vectors and item vectors for FAISS.
    np.save(
        DATA_DIR / "als_user_factors.npy",
        model.user_factors,
    )

    np.save(
        DATA_DIR / "als_item_factors.npy",
        model.item_factors,
    )

    # --------------------------------------------------------
    # Generate top-K recommendations for one sample user
    # --------------------------------------------------------

    # Use the first user, matching the sample user used in 07.
    sample_user_idx = 0
    sample_user_id = idx_user_map[
        str(sample_user_idx)
    ]

    # train_matrix[sample_user_idx] contains the items already
    # clicked by this user.
    #
    # filter_already_liked_items=True removes those items from
    # the recommendation result.
    recommended_items, recommended_scores = model.recommend(
        userid=sample_user_idx,
        user_items=train_matrix[sample_user_idx],
        N=TOP_K,
        filter_already_liked_items=True,
    )

    # --------------------------------------------------------
    # Verify that already-clicked items were removed
    # --------------------------------------------------------

    seen_items = set(
        train_matrix[sample_user_idx].indices
    )

    recommended_item_set = set(
        recommended_items
    )

    if not seen_items.isdisjoint(
        recommended_item_set
    ):
        raise ValueError(
            "ALS recommended an item already clicked by the user."
        )

    # --------------------------------------------------------
    # Build readable recommendation output
    # --------------------------------------------------------

    rows = []

    for rank, (item_idx, score) in enumerate(
        zip(recommended_items, recommended_scores),
        start=1,
    ):
        item_idx = int(item_idx)
        news_id = idx_item_map[str(item_idx)]

        rows.append(
            {
                "rank": rank,
                "item_idx": item_idx,
                "news_id": news_id,
                "title": title_map.get(
                    news_id,
                    "Title not found",
                ),
                "als_score": float(score),
            }
        )

    recommendation_table = pd.DataFrame(rows)

    print("\nSample user_idx:", sample_user_idx)
    print("Sample user_id:", sample_user_id)
    print(
        "Sample user's train clicks:",
        train_matrix[sample_user_idx].nnz,
    )

    print("\nALS top-10 recommendations:")
    print(
        recommendation_table.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        DATA_DIR / "als_model.npz",
    )

    print(
        "Saved:",
        DATA_DIR / "als_user_factors.npy",
    )

    print(
        "Saved:",
        DATA_DIR / "als_item_factors.npy",
    )


if __name__ == "__main__":
    main()