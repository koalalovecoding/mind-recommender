# ============================================================
# 16 - Full MIND Popularity and ALS Ranking Evaluation
#
# Evaluate both models on the same warm-start dev users.
# Metrics: Recall, NDCG, MRR, MAP, and Hit Rate.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"

K_VALUES = [10, 20, 40, 80]
MAX_K = max(K_VALUES)

# A smaller batch reduces memory usage during whole-catalog scoring.
BATCH_SIZE = 256


def evaluate_user(recommended, relevant, k):
    """Calculate five ranking metrics for one user."""

    hits = np.isin(
        recommended[:k],
        relevant,
    ).astype(np.float64)

    # Recall@K
    recall = hits.sum() / len(relevant)

    # NDCG@K
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

    ndcg = dcg / idcg

    # MRR@K
    hit_positions = np.flatnonzero(hits)

    mrr = (
        0.0
        if len(hit_positions) == 0
        else 1.0 / (hit_positions[0] + 1)
    )

    # MAP@K
    precision_at_rank = (
        np.cumsum(hits)
        / np.arange(1, len(hits) + 1)
    )

    map_k = (
        np.sum(precision_at_rank * hits)
        / min(len(relevant), k)
    )

    # Hit Rate@K
    hit_rate = float(hits.sum() > 0)

    return np.array(
        [recall, ndcg, mrr, map_k, hit_rate]
    )


def check_metrics():
    """Verify the metric functions with simple examples."""

    perfect = evaluate_user(
        recommended=np.array([1, 2]),
        relevant=np.array([1, 2]),
        k=2,
    )

    no_hit = evaluate_user(
        recommended=np.array([3, 4]),
        relevant=np.array([1, 2]),
        k=2,
    )

    assert np.allclose(perfect, 1.0)
    assert np.allclose(no_hit, 0.0)

    print("Metric sanity checks passed.")


def recommend_popular(
    user_idx,
    train_matrix,
    popularity_ranking,
):
    """Return the top-80 popular items not seen in train."""

    seen_items = set(
        train_matrix[user_idx].indices
    )

    recommendations = []

    for item_idx in popularity_ranking:

        if int(item_idx) not in seen_items:
            recommendations.append(item_idx)

        if len(recommendations) == MAX_K:
            break

    return np.asarray(
        recommendations,
        dtype=np.int64,
    )


def main():

    check_metrics()

    # --------------------------------------------------------
    # Load matrices, popularity ranking, and ALS model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Consistency checks
    # --------------------------------------------------------

    if train_matrix.shape != dev_matrix.shape:
        raise ValueError(
            "Train and dev matrix shapes do not match."
        )

    if model.user_factors.shape[0] != train_matrix.shape[0]:
        raise ValueError(
            "ALS user factors do not match matrix users."
        )

    if model.item_factors.shape[0] != train_matrix.shape[1]:
        raise ValueError(
            "ALS item factors do not match matrix items."
        )

    # Evaluate only users with at least one valid dev positive.
    users = np.flatnonzero(
        np.diff(dev_matrix.indptr) > 0
    )

    if len(users) == 0:
        raise ValueError(
            "No evaluable users were found."
        )

    print("Train matrix:", train_matrix.shape)
    print("Dev positive pairs:", dev_matrix.nnz)
    print("Evaluated users:", len(users))

    metric_sums = {
        "Popularity": {
            k: np.zeros(5)
            for k in K_VALUES
        },
        "ALS": {
            k: np.zeros(5)
            for k in K_VALUES
        },
    }

    # --------------------------------------------------------
    # Popularity evaluation
    # --------------------------------------------------------

    for position, user_idx in enumerate(
        users,
        start=1,
    ):
        recommendations = recommend_popular(
            user_idx,
            train_matrix,
            popularity_ranking,
        )

        relevant = dev_matrix[user_idx].indices

        for k in K_VALUES:
            metric_sums["Popularity"][k] += evaluate_user(
                recommendations,
                relevant,
                k,
            )

        if position % 50_000 == 0:
            print(
                f"Popularity evaluated: "
                f"{position}/{len(users)}"
            )

    # --------------------------------------------------------
    # ALS evaluation
    # --------------------------------------------------------

    for start in range(
        0,
        len(users),
        BATCH_SIZE,
    ):
        batch_users = users[
            start:start + BATCH_SIZE
        ]

        recommendations, _ = model.recommend(
            userid=batch_users,
            user_items=train_matrix[batch_users],
            N=MAX_K,
            filter_already_liked_items=True,
        )

        recommendations = np.atleast_2d(
            recommendations
        )

        for row, user_idx in enumerate(batch_users):

            relevant = dev_matrix[user_idx].indices

            for k in K_VALUES:
                metric_sums["ALS"][k] += evaluate_user(
                    recommendations[row],
                    relevant,
                    k,
                )

        completed = min(
            start + BATCH_SIZE,
            len(users),
        )

        if completed % 10_000 < BATCH_SIZE:
            print(
                f"ALS evaluated: "
                f"{completed}/{len(users)}"
            )

    # --------------------------------------------------------
    # Average metrics across all evaluated users
    # --------------------------------------------------------

    rows = []

    metric_names = [
        "Recall",
        "NDCG",
        "MRR",
        "MAP",
        "HitRate",
    ]

    for k in K_VALUES:
        for model_name in ["Popularity", "ALS"]:

            average_metrics = (
                metric_sums[model_name][k]
                / len(users)
            )

            row = {
                "Model": model_name,
                "K": k,
            }

            row.update(
                dict(
                    zip(
                        metric_names,
                        average_metrics,
                    )
                )
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    output_path = (
        DATA_DIR / "ranking_evaluation.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nFull MIND Popularity vs ALS:")
    print(
        results.round(6).to_string(
            index=False
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()