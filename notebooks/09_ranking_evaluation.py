# ============================================================
# 09 - Ranking Evaluation
#
# Evaluate Popularity and ALS on the same warm-start dev users.
# Metrics: Recall@K, NDCG@K, MRR@K, MAP@K, Hit Rate@K.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Evaluate both top-10 and top-40 recommendations.
K_VALUES = [10, 20, 40, 80]

BATCH_SIZE = 1024


# ------------------------------------------------------------
# Ranking metrics for one user
# ------------------------------------------------------------
def get_hits(recommended, relevant, k):
    """
    Return a binary array indicating whether each top-k item
    appears in the user's dev ground-truth items.
    """
    return np.isin(
        np.asarray(recommended)[:k],
        np.asarray(relevant),
    ).astype(np.float64)


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items recovered in the top-k list."""
    hits = get_hits(recommended, relevant, k)
    return hits.sum() / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Reward relevant items more when they appear near the top."""
    hits = get_hits(recommended, relevant, k)

    discounts = 1.0 / np.log2(
        np.arange(2, len(hits) + 2)
    )

    dcg = np.sum(hits * discounts)

    ideal_length = min(len(relevant), k)

    idcg = np.sum(
        1.0 / np.log2(
            np.arange(2, ideal_length + 2)
        )
    )

    return dcg / idcg


def reciprocal_rank_at_k(recommended, relevant, k):
    """Reciprocal rank of the first relevant recommendation."""
    hit_positions = np.flatnonzero(
        get_hits(recommended, relevant, k)
    )

    if len(hit_positions) == 0:
        return 0.0

    return 1.0 / (hit_positions[0] + 1)


def average_precision_at_k(recommended, relevant, k):
    """Average precision at ranks containing relevant items."""
    hits = get_hits(recommended, relevant, k)

    precision_at_rank = np.cumsum(hits) / np.arange(
        1,
        len(hits) + 1,
    )

    return np.sum(
        precision_at_rank * hits
    ) / min(len(relevant), k)


def hit_rate_at_k(recommended, relevant, k):
    """Return 1 if at least one relevant item appears in top-k."""
    return float(
        get_hits(recommended, relevant, k).sum() > 0
    )


def evaluate_user(recommended, relevant, k):
    """Calculate all five metrics for one user."""
    return [
        recall_at_k(recommended, relevant, k),
        ndcg_at_k(recommended, relevant, k),
        reciprocal_rank_at_k(recommended, relevant, k),
        average_precision_at_k(recommended, relevant, k),
        hit_rate_at_k(recommended, relevant, k),
    ]


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
def check_metrics():
    """
    Perfect ranking should produce all ones.
    No-hit ranking should produce all zeros.
    """
    perfect = evaluate_user(
        recommended=[1, 2],
        relevant=[1, 2],
        k=2,
    )

    no_hit = evaluate_user(
        recommended=[3, 4],
        relevant=[1, 2],
        k=2,
    )

    assert np.allclose(perfect, 1.0)
    assert np.allclose(no_hit, 0.0)

    print("Metric sanity checks passed.")


# ------------------------------------------------------------
# Popularity evaluation
# ------------------------------------------------------------
def evaluate_popularity(
    train_matrix,
    dev_matrix,
    popularity_ranking,
    users,
    k,
):
    """Evaluate globally popular unseen items for every dev user."""
    metric_rows = []

    for user_idx in users:

        # Items already clicked by this user in training.
        seen_items = set(
            train_matrix[user_idx].indices
        )

        recommendations = []

        # Follow the global popularity order, skipping seen items.
        for item_idx in popularity_ranking:

            if item_idx not in seen_items:
                recommendations.append(item_idx)

            if len(recommendations) == k:
                break

        relevant_items = dev_matrix[user_idx].indices

        metric_rows.append(
            evaluate_user(
                recommendations,
                relevant_items,
                k,
            )
        )

    return np.mean(metric_rows, axis=0)


# ------------------------------------------------------------
# ALS evaluation
# ------------------------------------------------------------
def evaluate_als(
    train_matrix,
    dev_matrix,
    model,
    users,
    k,
):
    """
    Generate ALS recommendations in batches and evaluate them.

    user_items is passed so implicit can filter training-seen items.
    """
    metric_rows = []

    for start in range(0, len(users), BATCH_SIZE):

        batch_users = users[
            start:start + BATCH_SIZE
        ]

        recommended_items, _ = model.recommend(
            userid=batch_users,
            user_items=train_matrix[batch_users],
            N=k,
            filter_already_liked_items=True,
        )

        # Keep the output two-dimensional even for a one-user batch.
        recommended_items = np.atleast_2d(
            recommended_items
        )

        for row, user_idx in enumerate(batch_users):

            relevant_items = dev_matrix[
                user_idx
            ].indices

            metric_rows.append(
                evaluate_user(
                    recommended_items[row],
                    relevant_items,
                    k,
                )
            )

    return np.mean(metric_rows, axis=0)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():

    check_metrics()

    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr()

    dev_matrix = load_npz(
        DATA_DIR / "dev_interactions.npz"
    ).tocsr()

    popularity_ranking = np.load(
        DATA_DIR / "popularity_ranking.npy"
    )

    model = AlternatingLeastSquares.load(
        DATA_DIR / "als_model.npz"
    )

    # Basic consistency checks.
    if train_matrix.shape != dev_matrix.shape:
        raise ValueError(
            "Train and dev matrix shapes do not match."
        )

    if model.user_factors.shape[0] != train_matrix.shape[0]:
        raise ValueError(
            "ALS user factors do not match the matrix users."
        )

    if model.item_factors.shape[0] != train_matrix.shape[1]:
        raise ValueError(
            "ALS item factors do not match the matrix items."
        )

    # Only evaluate users with at least one dev clicked item.
    dev_click_counts = np.diff(dev_matrix.indptr)

    users = np.flatnonzero(
        dev_click_counts > 0
    )

    if len(users) == 0:
        raise ValueError(
            "No evaluable users found in dev_interactions.npz."
        )

    print("Evaluated users:", len(users))

    # Diagnostic check:
    # Count how many globally popular items appear anywhere
    # among the positive items in the dev matrix.
    dev_positive_items = np.unique(dev_matrix.indices)

    for top_n in [10, 100, 1000]:

        overlap = np.intersect1d(
            popularity_ranking[:top_n],
            dev_positive_items,
        )

        print(
            f"Popularity top-{top_n} items appearing "
            f"in dev positives: {len(overlap)}"
        )

    # Evaluate both models for every requested K value.
    result_rows = []

    for k in K_VALUES:

        popularity_metrics = evaluate_popularity(
            train_matrix,
            dev_matrix,
            popularity_ranking,
            users,
            k,
        )

        als_metrics = evaluate_als(
            train_matrix,
            dev_matrix,
            model,
            users,
            k,
        )

        result_rows.append(
            {
                "Model": "Popularity",
                "K": k,
                "Recall": popularity_metrics[0],
                "NDCG": popularity_metrics[1],
                "MRR": popularity_metrics[2],
                "MAP": popularity_metrics[3],
                "HitRate": popularity_metrics[4],
            }
        )

        result_rows.append(
            {
                "Model": "ALS",
                "K": k,
                "Recall": als_metrics[0],
                "NDCG": als_metrics[1],
                "MRR": als_metrics[2],
                "MAP": als_metrics[3],
                "HitRate": als_metrics[4],
            }
        )

    results = pd.DataFrame(result_rows)

    output_path = (
        DATA_DIR / "ranking_evaluation.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nPopularity vs ALS:")
    print(
        results.round(6).to_string(
            index=False
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()