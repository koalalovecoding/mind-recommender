# ============================================================
# 10 - Two-Stage Pipeline and Evaluation
# ALS + FAISS top-100 -> filter seen -> rerank -> top-K
# Compare TwoStage with Popularity and ALS results from 09.
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "Install FAISS: pip install faiss-cpu"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

USER_IDX = 0

# Stage 1 returns 100 candidates.
CANDIDATE_K = 100

# Evaluate the reranked candidate list at these K values.
K_VALUES = [10, 20, 40, 80]

BATCH_SIZE = 1024

# Stage 2 reranking weights.
ALS_WEIGHT = 0.5
POPULARITY_WEIGHT = 0.5


# ------------------------------------------------------------
# Same ranking metrics as 09
# ------------------------------------------------------------
def hits(recommended, relevant, k):
    """
    Return a binary array showing whether each top-k item
    appears in the user's dev relevant items.
    """
    return np.isin(
        recommended[:k],
        relevant,
    ).astype(np.float64)


def evaluate_user(recommended, relevant, k):
    """Calculate all five ranking metrics for one user."""

    hit_array = hits(
        recommended,
        relevant,
        k,
    )

    # Recall@K
    recall = (
        hit_array.sum()
        / len(relevant)
    )

    # NDCG@K
    discounts = 1.0 / np.log2(
        np.arange(
            2,
            len(hit_array) + 2,
        )
    )

    dcg = np.sum(
        hit_array * discounts
    )

    ideal_length = min(
        len(relevant),
        k,
    )

    idcg = np.sum(
        1.0 / np.log2(
            np.arange(
                2,
                ideal_length + 2,
            )
        )
    )

    ndcg = dcg / idcg

    # MRR@K
    hit_positions = np.flatnonzero(
        hit_array
    )

    mrr = (
        0.0
        if len(hit_positions) == 0
        else 1.0 / (hit_positions[0] + 1)
    )

    # MAP@K
    precision_at_rank = (
        np.cumsum(hit_array)
        / np.arange(
            1,
            len(hit_array) + 1,
        )
    )

    map_k = (
        np.sum(
            precision_at_rank
            * hit_array
        )
        / min(len(relevant), k)
    )

    # Hit Rate@K
    hit_rate = float(
        hit_array.sum() > 0
    )

    return np.array(
        [
            recall,
            ndcg,
            mrr,
            map_k,
            hit_rate,
        ],
        dtype=np.float64,
    )


def check_metrics():
    """
    A perfect ranking should return all ones.
    A no-hit ranking should return all zeros.
    """

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

    assert np.allclose(
        perfect,
        1.0,
    )

    assert np.allclose(
        no_hit,
        0.0,
    )

    print(
        "Metric sanity checks passed."
    )


# ------------------------------------------------------------
# Stage 1 and Stage 2 helpers
# ------------------------------------------------------------
def min_max(values):
    """Scale one candidate feature to [0, 1]."""

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    value_range = (
        values.max()
        - values.min()
    )

    if value_range == 0:
        return np.zeros_like(
            values
        )

    return (
        values - values.min()
    ) / value_range


def filter_candidates(
    raw_items,
    seen_items,
):
    """
    Remove train-seen items and keep the first
    100 unseen FAISS candidates.
    """

    unseen_mask = ~np.isin(
        raw_items,
        seen_items,
    )

    candidates = raw_items[
        unseen_mask
    ][:CANDIDATE_K]

    if len(candidates) != CANDIDATE_K:
        raise ValueError(
            f"Expected {CANDIDATE_K} "
            f"candidates, got {len(candidates)}."
        )

    if len(
        np.unique(candidates)
    ) != CANDIDATE_K:
        raise ValueError(
            "Candidate items are not unique."
        )

    if np.intersect1d(
        candidates,
        seen_items,
    ).size:
        raise ValueError(
            "A train-seen item remains "
            "in the candidates."
        )

    return candidates


def rerank(
    candidate_items,
    user_vector,
    item_factors,
    popularity,
):
    """
    Stage 2:

    Recompute exact ALS scores for the 100 candidates,
    add a weak popularity feature, and rerank them.
    """

    exact_als_scores = (
        item_factors[candidate_items]
        @ user_vector
    )

    normalized_als = min_max(
        exact_als_scores
    )

    normalized_popularity = min_max(
        np.log1p(
            popularity[candidate_items]
        )
    )

    rerank_scores = (
        ALS_WEIGHT
        * normalized_als
        + POPULARITY_WEIGHT
        * normalized_popularity
    )

    rerank_order = np.argsort(
        -rerank_scores,
        kind="stable",
    )

    return (
        rerank_order,
        exact_als_scores,
        rerank_scores,
    )


# ------------------------------------------------------------
# Single-user example
# ------------------------------------------------------------
def save_sample_top10(
    index,
    train_matrix,
    user_factors,
    item_factors,
    popularity,
    idx_user_map,
    idx_item_map,
    title_map,
):
    """
    Generate and save the two-stage top-10
    recommendation table for one sample user.
    """

    user_vector = user_factors[
        USER_IDX
    ]

    seen_items = train_matrix[
        USER_IDX
    ].indices

    # Retrieve extra items because some FAISS results
    # may have already been clicked by the user.
    search_k = min(
        item_factors.shape[0],
        CANDIDATE_K
        + len(seen_items),
    )

    raw_scores, raw_items = index.search(
        user_vector[None, :],
        search_k,
    )

    raw_scores = raw_scores[0]
    raw_items = raw_items[0]

    # Verify that FAISS inner-product scores equal
    # the original ALS dot-product scores.
    exact_raw_scores = (
        item_factors[raw_items]
        @ user_vector
    )

    max_score_error = float(
        np.max(
            np.abs(
                raw_scores
                - exact_raw_scores
            )
        )
    )

    if not np.allclose(
        raw_scores,
        exact_raw_scores,
        atol=1e-5,
    ):
        raise ValueError(
            "FAISS scores do not match "
            "ALS dot products."
        )

    unseen_mask = ~np.isin(
        raw_items,
        seen_items,
    )

    candidate_items = filter_candidates(
        raw_items,
        seen_items,
    )

    candidate_faiss_scores = raw_scores[
        unseen_mask
    ][:CANDIDATE_K]

    (
        rerank_order,
        exact_als_scores,
        rerank_scores,
    ) = rerank(
        candidate_items,
        user_vector,
        item_factors,
        popularity,
    )

    final_positions = rerank_order[
        :10
    ]

    rows = []

    for final_rank, position in enumerate(
        final_positions,
        start=1,
    ):
        item_idx = int(
            candidate_items[position]
        )

        news_id = idx_item_map[
            str(item_idx)
        ]

        rows.append(
            {
                "final_rank": final_rank,

                # Rank before Stage 2 reranking.
                "retrieval_rank": int(
                    position + 1
                ),

                "item_idx": item_idx,
                "news_id": news_id,

                "title": title_map.get(
                    news_id,
                    "Title not found",
                ),

                "faiss_score": float(
                    candidate_faiss_scores[
                        position
                    ]
                ),

                "als_score": float(
                    exact_als_scores[
                        position
                    ]
                ),

                "popularity": int(
                    popularity[item_idx]
                ),

                "rerank_score": float(
                    rerank_scores[
                        position
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    output_path = (
        DATA_DIR
        / "two_stage_top10.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSample user:",
        USER_IDX,
        idx_user_map[str(USER_IDX)],
    )

    print(
        "seen items:",
        len(seen_items),
    )

    print(
        "raw candidates:",
        len(raw_items),
    )

    print(
        "removed seen items:",
        int(
            np.isin(
                raw_items,
                seen_items,
            ).sum()
        ),
    )

    print(
        "unseen candidates:",
        len(candidate_items),
    )

    print(
        "maximum FAISS score error:",
        max_score_error,
    )

    print(
        "reranked positions changed:",
        int(
            np.sum(
                rerank_order
                != np.arange(
                    CANDIDATE_K
                )
            )
        ),
    )

    print(
        "\nFinal top-10:"
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        output_path,
    )


# ------------------------------------------------------------
# Evaluate all warm-start dev users
# ------------------------------------------------------------
def evaluate_two_stage(
    index,
    train_matrix,
    dev_matrix,
    user_factors,
    item_factors,
    popularity,
    users,
):
    """
    Run the full two-stage pipeline once per user.

    The same reranked top-100 list is used to evaluate
    K = 10, 20, 40, and 80.
    """

    metric_sums = {
        k: np.zeros(
            5,
            dtype=np.float64,
        )
        for k in K_VALUES
    }

    retrieval_recall_sum = 0.0
    num_items = item_factors.shape[0]

    for start in range(
        0,
        len(users),
        BATCH_SIZE,
    ):
        batch_users = users[
            start:start + BATCH_SIZE
        ]

        batch_train = train_matrix[
            batch_users
        ]

        batch_vectors = np.ascontiguousarray(
            user_factors[
                batch_users
            ],
            dtype=np.float32,
        )

        # Use the largest seen-item count in the batch.
        # This guarantees that every user can retain
        # 100 unseen candidates after filtering.
        max_seen = int(
            np.diff(
                batch_train.indptr
            ).max()
        )

        search_k = min(
            num_items,
            CANDIDATE_K
            + max_seen,
        )

        _, batch_items = index.search(
            batch_vectors,
            search_k,
        )

        for row, user_idx in enumerate(
            batch_users
        ):
            seen_items = batch_train[
                row
            ].indices

            candidate_items = filter_candidates(
                batch_items[row],
                seen_items,
            )

            (
                rerank_order,
                _,
                _,
            ) = rerank(
                candidate_items,
                batch_vectors[row],
                item_factors,
                popularity,
            )

            recommendations = candidate_items[
                rerank_order
            ]

            relevant_items = dev_matrix[
                user_idx
            ].indices

            # Stage-1 retrieval quality before reranking.
            retrieval_hits = hits(
                candidate_items,
                relevant_items,
                CANDIDATE_K,
            )

            retrieval_recall_sum += (
                retrieval_hits.sum()
                / len(relevant_items)
            )

            # Use the same top-100 result for all K values.
            for k in K_VALUES:
                metric_sums[k] += evaluate_user(
                    recommendations,
                    relevant_items,
                    k,
                )

        evaluated_count = min(
            start + BATCH_SIZE,
            len(users),
        )

        print(
            f"Evaluated "
            f"{evaluated_count}/"
            f"{len(users)} users"
        )

    result_rows = []

    for k in K_VALUES:
        (
            recall,
            ndcg,
            mrr,
            map_k,
            hit_rate,
        ) = (
            metric_sums[k]
            / len(users)
        )

        result_rows.append(
            {
                "Model": "TwoStage",
                "K": k,
                "Recall": recall,
                "NDCG": ndcg,
                "MRR": mrr,
                "MAP": map_k,
                "HitRate": hit_rate,
            }
        )

    retrieval_recall = (
        retrieval_recall_sum
        / len(users)
    )

    return (
        pd.DataFrame(
            result_rows
        ),
        retrieval_recall,
    )


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():

    check_metrics()

    # --------------------------------------------------------
    # Load matrices and model outputs
    # --------------------------------------------------------

    train_matrix = load_npz(
        DATA_DIR
        / "train_interactions.npz"
    ).tocsr()

    dev_matrix = load_npz(
        DATA_DIR
        / "dev_interactions.npz"
    ).tocsr()

    user_factors = np.ascontiguousarray(
        np.load(
            DATA_DIR
            / "als_user_factors.npy"
        ),
        dtype=np.float32,
    )

    item_factors = np.ascontiguousarray(
        np.load(
            DATA_DIR
            / "als_item_factors.npy"
        ),
        dtype=np.float32,
    )

    popularity = np.load(
        DATA_DIR
        / "popularity_scores.npy"
    )

    with open(
        DATA_DIR
        / "idx_user_map.json",
        encoding="utf-8",
    ) as file:
        idx_user_map = json.load(
            file
        )

    with open(
        DATA_DIR
        / "idx_item_map.json",
        encoding="utf-8",
    ) as file:
        idx_item_map = json.load(
            file
        )

    news = pd.read_parquet(
        DATA_DIR
        / "news.parquet",
        columns=[
            "news_id",
            "title",
        ],
    ).drop_duplicates(
        subset=["news_id"]
    )

    title_map = dict(
        zip(
            news["news_id"],
            news["title"],
        )
    )

    # --------------------------------------------------------
    # Consistency checks
    # --------------------------------------------------------

    if train_matrix.shape != dev_matrix.shape:
        raise ValueError(
            "Train and dev matrix "
            "shapes do not match."
        )

    num_users, num_items = (
        train_matrix.shape
    )

    if (
        user_factors.shape[0]
        != num_users
    ):
        raise ValueError(
            "User factors do not match "
            "the matrix users."
        )

    if (
        item_factors.shape[0]
        != num_items
    ):
        raise ValueError(
            "Item factors do not match "
            "the matrix items."
        )

    if (
        user_factors.shape[1]
        != item_factors.shape[1]
    ):
        raise ValueError(
            "User and item factor "
            "dimensions do not match."
        )

    if len(popularity) != num_items:
        raise ValueError(
            "Popularity scores do not "
            "match the item space."
        )

    if (
        len(idx_user_map)
        != num_users
    ):
        raise ValueError(
            "User mapping does not "
            "match the user space."
        )

    if (
        len(idx_item_map)
        != num_items
    ):
        raise ValueError(
            "Item mapping does not "
            "match the item space."
        )

    if CANDIDATE_K < max(K_VALUES):
        raise ValueError(
            "CANDIDATE_K must be at "
            "least max(K_VALUES)."
        )

    # --------------------------------------------------------
    # Build the FAISS index
    # --------------------------------------------------------

    # IndexFlatIP performs exact maximum
    # inner-product search.
    index = faiss.IndexFlatIP(
        item_factors.shape[1]
    )

    index.add(
        item_factors
    )

    if index.ntotal != num_items:
        raise ValueError(
            "FAISS index does not "
            "contain every item."
        )

    # --------------------------------------------------------
    # Single-user demonstration
    # --------------------------------------------------------

    save_sample_top10(
        index,
        train_matrix,
        user_factors,
        item_factors,
        popularity,
        idx_user_map,
        idx_item_map,
        title_map,
    )

    # --------------------------------------------------------
    # Full dev evaluation
    # --------------------------------------------------------

    # Same users as 09:
    # users with at least one dev positive item.
    dev_positive_counts = np.diff(
        dev_matrix.indptr
    )

    users = np.flatnonzero(
        dev_positive_counts > 0
    )

    if len(users) == 0:
        raise ValueError(
            "No evaluable dev users found."
        )

    print(
        "\nEvaluated users:",
        len(users),
    )

    (
        two_stage_results,
        retrieval_recall,
    ) = evaluate_two_stage(
        index,
        train_matrix,
        dev_matrix,
        user_factors,
        item_factors,
        popularity,
        users,
    )

    # --------------------------------------------------------
    # Save TwoStage results
    # --------------------------------------------------------

    two_stage_path = (
        DATA_DIR
        / "two_stage_evaluation.csv"
    )

    two_stage_results.to_csv(
        two_stage_path,
        index=False,
    )

    diagnostics = pd.DataFrame(
        [
            {
                "EvaluatedUsers": len(users),
                "CandidateK": CANDIDATE_K,
                "RetrievalRecallAt100": (
                    retrieval_recall
                ),
                "ALSWeight": ALS_WEIGHT,
                "PopularityWeight": (
                    POPULARITY_WEIGHT
                ),
            }
        ]
    )

    diagnostics_path = (
        DATA_DIR
        / "two_stage_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Merge with Popularity and ALS results from 09
    # --------------------------------------------------------

    baseline_path = (
        DATA_DIR
        / "ranking_evaluation.csv"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            "Run "
            "notebooks/09_ranking_evaluation.py "
            "first."
        )

    baseline_results = pd.read_csv(
        baseline_path
    )

    baseline_results = baseline_results[
        baseline_results["K"].isin(
            K_VALUES
        )
    ]

    comparison = pd.concat(
        [
            baseline_results,
            two_stage_results,
        ],
        ignore_index=True,
    )

    model_order = {
        "Popularity": 0,
        "ALS": 1,
        "TwoStage": 2,
    }

    comparison["_order"] = (
        comparison["Model"].map(
            model_order
        )
    )

    comparison = comparison.sort_values(
        [
            "K",
            "_order",
        ]
    ).drop(
        columns="_order"
    )

    comparison_path = (
        DATA_DIR
        / "ranking_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\nTwo-stage evaluation:"
    )

    print(
        two_stage_results.round(
            6
        ).to_string(
            index=False
        )
    )

    print(
        f"\nRetrieval "
        f"Recall@{CANDIDATE_K}: "
        f"{retrieval_recall:.6f}"
    )

    print(
        "\nPopularity vs ALS vs TwoStage:"
    )

    print(
        comparison.round(
            6
        ).to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        two_stage_path,
    )

    print(
        "Saved:",
        diagnostics_path,
    )

    print(
        "Saved:",
        comparison_path,
    )


if __name__ == "__main__":
    main()