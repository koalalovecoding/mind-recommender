# ============================================================
# Phase 2 Part C - Step 10: Popularity Baseline
#
# Goal:
# Build the simplest recommendation baseline on MIND-small.
# The model recommends the news items with the most training clicks.
#
# Important rules:
# 1. Popularity is calculated from TRAIN data only.
# 2. Dev data is not used here, which avoids data leakage.
# 3. Items already clicked by the user in train are filtered out.
#
# Inputs:
#   ../data/processed/train_interactions.npz
#   ../data/processed/idx_user_map.json
#   ../data/processed/idx_item_map.json
#   ../data/processed/news.parquet
#
# Outputs:
#   ../data/processed/popularity_scores.npy
#   ../data/processed/popularity_ranking.npy
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOP_K = 10


# ------------------------------------------------------------
# Calculate item popularity
# ------------------------------------------------------------
def calculate_item_popularity(train_matrix):
    """Return the number of training users who clicked each item."""

    # train_matrix has shape:
    #
    #     number of users x number of items
    #
    # The matrix created in Step 8 is binary:
    #
    #     train_matrix[u, i] = 1
    #
    # means user u clicked item i at least once in the training data.
    #
    # Therefore, summing each item column over all users gives the
    # number of unique training users who clicked that news item.
    #
    # train_matrix.sum(axis=0) returns an object with shape:
    #
    #     (1, number of items)
    #
    # np.asarray(...).ravel() converts it into a one-dimensional
    # NumPy array with shape:
    #
    #     (number of items,)
    item_popularity = np.asarray(
        train_matrix.sum(axis=0)
    ).ravel()

    return item_popularity


# ------------------------------------------------------------
# Recommend popular unseen items to one user
# ------------------------------------------------------------
def recommend_popular_items(
    user_idx,
    train_matrix,
    popularity_ranking,
    item_popularity,
    k=10,
):
    """
    Return the top-k most popular items that the user has not
    already clicked in the training set.
    """

    # Make sure the requested user index exists in the matrix.
    if user_idx < 0 or user_idx >= train_matrix.shape[0]:
        raise IndexError(
            f"user_idx must be between 0 and "
            f"{train_matrix.shape[0] - 1}."
        )

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # train_matrix is stored in CSR format.
    #
    # For one CSR row, .indices directly returns the column indices
    # of all nonzero entries.
    #
    # These item indices correspond to news items that the user
    # already clicked in the training data.
    seen_items = set(
        train_matrix[user_idx].indices
    )

    recommended_items = []

    # popularity_ranking contains every item index ordered from
    # the most popular item to the least popular item.
    #
    # The popularity model gives every user the same global ranking.
    # The only user-specific operation is filtering out items that
    # this user has already clicked.
    for item_idx in popularity_ranking:

        if item_idx not in seen_items:
            recommended_items.append(item_idx)

        # Stop as soon as k unseen items have been collected.
        if len(recommended_items) == k:
            break

    # Convert the Python list into a NumPy integer array.
    # This makes it easier to use the indices for vectorized lookup.
    recommended_items = np.asarray(
        recommended_items,
        dtype=np.int64,
    )

    # Retrieve the popularity score for each recommended item.
    # The score is the number of training users who clicked the item.
    recommended_scores = item_popularity[
        recommended_items
    ]

    return recommended_items, recommended_scores


# ------------------------------------------------------------
# Convert matrix item indices into readable news information
# ------------------------------------------------------------
def build_recommendation_table(
    item_indices,
    scores,
    idx_item_map,
    title_map,
):
    """
    Create a readable recommendation table containing:

    rank
    item_idx
    raw MIND news_id
    news title
    popularity score
    """

    rows = []

    for rank, (item_idx, score) in enumerate(
        zip(item_indices, scores),
        start=1,
    ):
        # JSON object keys are always strings after json.load().
        #
        # idx_item_map was saved in the following form:
        #
        #     "15" -> "N12345"
        #
        # Therefore, the integer item index must be converted to
        # a string before using it as a dictionary key.
        news_id = idx_item_map[
            str(item_idx)
        ]

        rows.append(
            {
                "rank": rank,
                "item_idx": int(item_idx),
                "news_id": news_id,

                # Some items may theoretically have missing title
                # metadata. title_map.get() prevents a KeyError.
                "title": title_map.get(
                    news_id,
                    "Title not found",
                ),

                # Convert the NumPy score into a normal Python int
                # so the printed and saved result is easier to read.
                "popularity_score": int(score),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():

    # --------------------------------------------------------
    # Load the train interaction matrix
    # --------------------------------------------------------

    # This is the binary CSR user-item matrix created by:
    #
    #     06_user_item_interaction_matrix.py
    #
    # Each nonzero entry represents one positive train interaction.
    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr()

    # --------------------------------------------------------
    # Load reverse ID mappings
    # --------------------------------------------------------

    # idx_user_map:
    #
    #     integer user_idx -> raw MIND user_id
    #
    # Example:
    #
    #     "0" -> "U12345"
    with open(
        DATA_DIR / "idx_user_map.json",
        "r",
    ) as f:
        idx_user_map = json.load(f)

    # idx_item_map:
    #
    #     integer item_idx -> raw MIND news_id
    #
    # Example:
    #
    #     "15" -> "N67890"
    with open(
        DATA_DIR / "idx_item_map.json",
        "r",
    ) as f:
        idx_item_map = json.load(f)

    # --------------------------------------------------------
    # Load news titles
    # --------------------------------------------------------

    # news.parquet is the compact news metadata table.
    #
    # We do not need to load train_with_news.parquet here because
    # that file contains millions of interaction rows. Loading the
    # smaller news table is enough to display recommendation titles.
    news_df = pd.read_parquet(
        DATA_DIR / "news.parquet"
    )

    # Keep one row for every news ID.
    news_title_df = news_df[
        ["news_id", "title"]
    ].drop_duplicates(
        subset=["news_id"]
    )

    # Create:
    #
    #     raw news_id -> news title
    #
    # This allows the final recommendation output to show readable
    # titles instead of only matrix item indices.
    title_map = dict(
        zip(
            news_title_df["news_id"],
            news_title_df["title"],
        )
    )

    # --------------------------------------------------------
    # Check matrix and mapping consistency
    # --------------------------------------------------------

    # The number of matrix rows should equal the number of users
    # in the reverse user mapping.
    if train_matrix.shape[0] != len(idx_user_map):
        raise ValueError(
            "The number of train matrix rows does not match "
            "the number of users in idx_user_map."
        )

    # The number of matrix columns should equal the number of
    # items in the reverse item mapping.
    if train_matrix.shape[1] != len(idx_item_map):
        raise ValueError(
            "The number of train matrix columns does not match "
            "the number of items in idx_item_map."
        )

    # --------------------------------------------------------
    # Calculate popularity scores
    # --------------------------------------------------------

    item_popularity = calculate_item_popularity(
        train_matrix
    )

    # np.argsort normally sorts from the smallest value to the
    # largest value.
    #
    # Sorting -item_popularity therefore produces descending order:
    #
    #     most popular item -> least popular item
    #
    # kind="stable" means that tied items keep a deterministic order.
    popularity_ranking = np.argsort(
        -item_popularity,
        kind="stable",
    )

    # --------------------------------------------------------
    # Save popularity results for later scripts
    # --------------------------------------------------------

    # popularity_scores.npy:
    #
    #     popularity_scores[item_idx]
    #
    # returns the popularity score of that item.
    np.save(
        DATA_DIR / "popularity_scores.npy",
        item_popularity,
    )

    # popularity_ranking.npy:
    #
    # contains all item indices sorted from highest popularity
    # to lowest popularity.
    #
    # This can be reused later by the evaluation script without
    # recalculating the global item ranking.
    np.save(
        DATA_DIR / "popularity_ranking.npy",
        popularity_ranking,
    )

    # --------------------------------------------------------
    # Print basic diagnostic information
    # --------------------------------------------------------

    print(
        "train matrix shape:",
        train_matrix.shape,
    )

    print(
        "train matrix nnz:",
        train_matrix.nnz,
    )

    # Count how many news items received at least one training click.
    print(
        "items with at least one click:",
        np.count_nonzero(item_popularity),
    )

    # Show the popularity score of the most popular news item.
    print(
        "maximum popularity score:",
        int(item_popularity.max()),
    )

    # --------------------------------------------------------
    # Display global top-10 popular items
    # --------------------------------------------------------

    # These are the globally most popular items before filtering
    # any individual user's previous clicks.
    global_top_items = popularity_ranking[
        :TOP_K
    ]

    global_top_scores = item_popularity[
        global_top_items
    ]

    global_top_table = build_recommendation_table(
        item_indices=global_top_items,
        scores=global_top_scores,
        idx_item_map=idx_item_map,
        title_map=title_map,
    )

    print(
        "\nGlobal top-10 popular news items:"
    )

    print(
        global_top_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select one sample user
    # --------------------------------------------------------

    # For a CSR matrix:
    #
    #     train_matrix.indptr
    #
    # stores the boundaries of each row's nonzero entries.
    #
    # np.diff(train_matrix.indptr) gives the number of nonzero
    # entries in each user row.
    #
    # Here, that is the number of unique news items clicked by
    # each user in the training data.
    clicks_per_user = np.diff(
        train_matrix.indptr
    )

    # Find users who have at least one training interaction.
    nonempty_users = np.flatnonzero(
        clicks_per_user > 0
    )

    if len(nonempty_users) == 0:
        raise ValueError(
            "The training matrix contains no positive interactions."
        )

    # Use the first nonempty user for the sample output.
    sample_user_idx = int(
        nonempty_users[0]
    )

    # JSON keys are strings, so convert the integer user index
    # before looking up the original MIND user ID.
    sample_user_id = idx_user_map[
        str(sample_user_idx)
    ]

    # --------------------------------------------------------
    # Generate sample-user top-10 recommendations
    # --------------------------------------------------------

    recommended_items, recommended_scores = (
        recommend_popular_items(
            user_idx=sample_user_idx,
            train_matrix=train_matrix,
            popularity_ranking=popularity_ranking,
            item_popularity=item_popularity,
            k=TOP_K,
        )
    )

    recommendation_table = build_recommendation_table(
        item_indices=recommended_items,
        scores=recommended_scores,
        idx_item_map=idx_item_map,
        title_map=title_map,
    )

    # --------------------------------------------------------
    # Confirm that seen-item filtering worked
    # --------------------------------------------------------

    # Get all training items clicked by the sample user.
    sample_seen_items = set(
        train_matrix[
            sample_user_idx
        ].indices
    )

    # Convert recommended items to a set for comparison.
    sample_recommended_items = set(
        recommended_items
    )

    # The two sets must not overlap.
    #
    # If they overlap, the recommender incorrectly included an
    # item that the user already clicked in training.
    if not sample_seen_items.isdisjoint(
        sample_recommended_items
    ):
        raise ValueError(
            "The recommendation list contains an item that "
            "the sample user already clicked in train."
        )

    # --------------------------------------------------------
    # Print sample-user recommendation output
    # --------------------------------------------------------

    print(
        "\nSample user_idx:",
        sample_user_idx,
    )

    print(
        "Sample user_id:",
        sample_user_id,
    )

    print(
        "Sample user's train clicks:",
        int(clicks_per_user[sample_user_idx]),
    )

    print(
        "\nPopularity top-10 recommendations "
        "for the sample user:"
    )

    print(
        recommendation_table.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        DATA_DIR / "popularity_scores.npy",
    )

    print(
        "Saved:",
        DATA_DIR / "popularity_ranking.npy",
    )


# This condition means main() runs when the script is executed:
#
#     python 07_popularity_baseline.py
#
# If another script imports one of the functions from this file,
# main() will not run automatically.
if __name__ == "__main__":
    main()