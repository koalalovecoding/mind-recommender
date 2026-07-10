# ============================================================
# 17 - Full MIND Two-Stage Pipeline
#
# ALS factors
# -> FAISS top-100 retrieval
# -> filter train-seen items
# -> ALS + popularity reranking
# -> evaluate K = 10, 20, 40, 80
# -> compare Popularity, ALS, and TwoStage
# ============================================================

import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"

NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MINDlarge_train"
    / "news.tsv"
)

# Stage 1 retrieves 100 candidates.
CANDIDATE_K = 100

# Evaluate the reranked list at four values of K.
K_VALUES = [10, 20, 40, 80]

# The readable sample output still contains only top-10.
SAMPLE_TOP_K = 10

BATCH_SIZE = 256

# Heuristic reranking weights.
ALS_WEIGHT = 0.99
POPULARITY_WEIGHT = 0.01

# Change to 1.0 and 0.0 to run the ALS-only sanity check.
# ALS_WEIGHT = 1.0
# POPULARITY_WEIGHT = 0.0

SAMPLE_USER_IDX = 0


def min_max(values):
    """Scale one candidate feature to [0, 1]."""

    value_range = values.max() - values.min()

    if value_range == 0:
        return np.zeros_like(values)

    return (
        values - values.min()
    ) / value_range


def get_unseen_candidates(
    items,
    scores,
    seen_items,
):
    """Remove train-seen items and keep top-100 unseen items."""

    unseen_mask = ~np.isin(
        items,
        seen_items,
    )

    items = items[
        unseen_mask
    ][:CANDIDATE_K]

    scores = scores[
        unseen_mask
    ][:CANDIDATE_K]

    if len(items) != CANDIDATE_K:
        raise ValueError(
            "Could not retrieve 100 unseen candidates."
        )

    if len(np.unique(items)) != CANDIDATE_K:
        raise ValueError(
            "Candidate items contain duplicates."
        )

    if np.intersect1d(
        items,
        seen_items,
    ).size:
        raise ValueError(
            "A train-seen item remains in the candidates."
        )

    return items, scores


def rerank(
    candidate_items,
    user_vector,
    item_factors,
    popularity,
):
    """Combine normalized ALS score and log-popularity."""

    als_scores = (
        item_factors[candidate_items]
        @ user_vector
    )

    popularity_scores = np.log1p(
        popularity[candidate_items]
    )

    rerank_scores = (
        ALS_WEIGHT
        * min_max(als_scores)
        + POPULARITY_WEIGHT
        * min_max(popularity_scores)
    )

    order = np.argsort(
        -rerank_scores,
        kind="stable",
    )

    return als_scores, rerank_scores, order


def evaluate_user(
    recommended,
    relevant,
    k,
):
    """
    Calculate Recall@K, NDCG@K, MRR@K,
    MAP@K, and HitRate@K for one user.
    """

    hits = np.isin(
        recommended[:k],
        relevant,
    ).astype(np.float64)

    # Recall@K
    recall = (
        hits.sum()
        / len(relevant)
    )

    # NDCG@K
    discounts = (
        1.0
        / np.log2(
            np.arange(
                2,
                len(hits) + 2,
            )
        )
    )

    dcg = np.sum(
        hits * discounts
    )

    ideal_length = min(
        len(relevant),
        k,
    )

    idcg = np.sum(
        discounts[:ideal_length]
    )

    ndcg = dcg / idcg

    # MRR@K
    hit_positions = np.flatnonzero(
        hits
    )

    mrr = (
        0.0
        if len(hit_positions) == 0
        else 1.0 / (hit_positions[0] + 1)
    )

    # MAP@K
    precision_at_rank = (
        np.cumsum(hits)
        / np.arange(
            1,
            len(hits) + 1,
        )
    )

    map_k = (
        np.sum(
            precision_at_rank
            * hits
        )
        / ideal_length
    )

    # Hit Rate@K
    hit_rate = float(
        hits.sum() > 0
    )

    return np.array(
        [
            recall,
            ndcg,
            mrr,
            map_k,
            hit_rate,
        ]
    )


def build_model_comparison(
    two_stage_evaluation,
    evaluated_users,
):
    """
    Combine Step 16 Popularity/ALS results with
    the current TwoStage results at all K values.
    """

    baseline_path = (
        DATA_DIR
        / "ranking_evaluation.csv"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            "Run Step 16 before Step 17. "
            "ranking_evaluation.csv was not found."
        )

    baseline = pd.read_csv(
        baseline_path
    )

    # Keep Popularity and ALS at K = 10, 20, 40, 80.
    baseline = baseline[
        baseline["K"].isin(K_VALUES)
        & baseline["Model"].isin(
            ["Popularity", "ALS"]
        )
    ].copy()

    expected_rows = (
        len(K_VALUES) * 2
    )

    if len(baseline) != expected_rows:
        raise ValueError(
            "Step 16 results must contain "
            "Popularity and ALS at "
            "K = 10, 20, 40, and 80."
        )

    baseline.insert(
        2,
        "EvaluatedUsers",
        evaluated_users,
    )

    # Candidate Recall is only defined for TwoStage.
    baseline["CandidateRecall@100"] = np.nan
    baseline["ALSWeight"] = np.nan
    baseline["PopularityWeight"] = np.nan

    baseline = baseline[
        [
            "Model",
            "K",
            "EvaluatedUsers",
            "Recall",
            "NDCG",
            "MRR",
            "MAP",
            "HitRate",
            "CandidateRecall@100",
            "ALSWeight",
            "PopularityWeight",
        ]
    ]

    comparison = pd.concat(
        [
            baseline,
            two_stage_evaluation,
        ],
        ignore_index=True,
    )

    model_order = {
        "Popularity": 0,
        "ALS": 1,
        "TwoStageHeuristic": 2,
    }

    comparison["_model_order"] = (
        comparison["Model"]
        .map(model_order)
    )

    comparison = (
        comparison
        .sort_values(
            [
                "K",
                "_model_order",
            ]
        )
        .drop(
            columns="_model_order"
        )
        .reset_index(
            drop=True
        )
    )

    # When the reranker uses only ALS, the TwoStage results
    # should reproduce direct ALS at every K.
    if (
        np.isclose(
            ALS_WEIGHT,
            1.0,
        )
        and np.isclose(
            POPULARITY_WEIGHT,
            0.0,
        )
    ):
        metric_columns = [
            "Recall",
            "NDCG",
            "MRR",
            "MAP",
            "HitRate",
        ]

        als_metrics = (
            comparison[
                comparison["Model"] == "ALS"
            ]
            .sort_values("K")[
                metric_columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        two_stage_metrics = (
            comparison[
                comparison["Model"]
                == "TwoStageHeuristic"
            ]
            .sort_values("K")[
                metric_columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.allclose(
            als_metrics,
            two_stage_metrics,
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError(
                "ALS-only TwoStage results do not "
                "match direct ALS results."
            )

        print(
            "ALS-only pipeline sanity check passed."
        )

    return comparison


def main():

    # --------------------------------------------------------
    # Load matrices, factors, popularity, and metadata
    # --------------------------------------------------------

    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr()

    dev_matrix = load_npz(
        DATA_DIR / "dev_interactions.npz"
    ).tocsr()

    user_factors = np.load(
        DATA_DIR / "als_user_factors.npy",
        mmap_mode="r",
    )

    item_factors = np.ascontiguousarray(
        np.load(
            DATA_DIR
            / "als_item_factors.npy"
        ),
        dtype=np.float32,
    )

    popularity = np.load(
        DATA_DIR / "popularity_scores.npy"
    )

    with open(
        DATA_DIR / "idx_user_map.json",
        encoding="utf-8",
    ) as file:
        idx_user_map = json.load(file)

    with open(
        DATA_DIR / "idx_item_map.json",
        encoding="utf-8",
    ) as file:
        idx_item_map = json.load(file)

    news = pd.read_csv(
        NEWS_PATH,
        sep="\t",
        header=None,
        usecols=[0, 3],
        names=[
            "news_id",
            "title",
        ],
    )

    title_map = dict(
        zip(
            news["news_id"],
            news["title"],
        )
    )

    if train_matrix.shape != dev_matrix.shape:
        raise ValueError(
            "Train and dev matrix shapes do not match."
        )

    if user_factors.shape[0] != train_matrix.shape[0]:
        raise ValueError(
            "User factors do not match matrix users."
        )

    if item_factors.shape[0] != train_matrix.shape[1]:
        raise ValueError(
            "Item factors do not match matrix items."
        )

    # --------------------------------------------------------
    # Build FAISS IndexFlatIP
    # --------------------------------------------------------

    start = time.perf_counter()

    index = faiss.IndexFlatIP(
        item_factors.shape[1]
    )

    index.add(
        item_factors
    )

    index_build_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Sample-user candidates and final top-10
    # --------------------------------------------------------

    user_idx = SAMPLE_USER_IDX

    user_vector = np.ascontiguousarray(
        user_factors[user_idx],
        dtype=np.float32,
    )

    seen_items = train_matrix[
        user_idx
    ].indices

    search_k = min(
        train_matrix.shape[1],
        CANDIDATE_K + len(seen_items),
    )

    raw_scores, raw_items = index.search(
        user_vector[None, :],
        search_k,
    )

    raw_scores = raw_scores[0]
    raw_items = raw_items[0]

    # FAISS inner-product scores must equal ALS dot products.
    direct_scores = (
        item_factors[raw_items]
        @ user_vector
    )

    max_score_error = float(
        np.max(
            np.abs(
                raw_scores
                - direct_scores
            )
        )
    )

    if not np.allclose(
        raw_scores,
        direct_scores,
        atol=1e-5,
    ):
        raise ValueError(
            "FAISS scores do not match ALS dot products."
        )

    (
        candidate_items,
        candidate_faiss_scores,
    ) = get_unseen_candidates(
        raw_items,
        raw_scores,
        seen_items,
    )

    (
        als_scores,
        rerank_scores,
        order,
    ) = rerank(
        candidate_items,
        user_vector,
        item_factors,
        popularity,
    )

    candidate_rows = []

    for retrieval_rank, item_idx in enumerate(
        candidate_items,
        start=1,
    ):
        item_idx = int(
            item_idx
        )

        news_id = idx_item_map[
            str(item_idx)
        ]

        position = (
            retrieval_rank - 1
        )

        candidate_rows.append(
            {
                "retrieval_rank": retrieval_rank,
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
                    als_scores[
                        position
                    ]
                ),
                "popularity": int(
                    popularity[
                        item_idx
                    ]
                ),
                "rerank_score": float(
                    rerank_scores[
                        position
                    ]
                ),
            }
        )

    candidate_table = pd.DataFrame(
        candidate_rows
    )

    # The readable sample output remains top-10.
    final_table = candidate_table.iloc[
        order[:SAMPLE_TOP_K]
    ].copy()

    final_table.insert(
        0,
        "final_rank",
        np.arange(
            1,
            SAMPLE_TOP_K + 1,
        ),
    )

    final_table.insert(
        1,
        "user_idx",
        user_idx,
    )

    final_table.insert(
        2,
        "user_id",
        idx_user_map[
            str(user_idx)
        ],
    )

    candidate_table.to_csv(
        DATA_DIR
        / "two_stage_candidates_sample.csv",
        index=False,
    )

    final_table.to_csv(
        DATA_DIR
        / "two_stage_top10.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Evaluate all users with valid dev positives
    # --------------------------------------------------------

    users = np.flatnonzero(
        np.diff(
            dev_matrix.indptr
        ) > 0
    )

    # Store separate metric sums for each K.
    metric_sums = {
        k: np.zeros(
            5,
            dtype=np.float64,
        )
        for k in K_VALUES
    }

    candidate_recall_sum = 0.0

    retrieval_time = 0.0
    reranking_time = 0.0

    total_start = time.perf_counter()

    for start_idx in range(
        0,
        len(users),
        BATCH_SIZE,
    ):
        batch_users = users[
            start_idx:
            start_idx + BATCH_SIZE
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

        max_seen = int(
            np.diff(
                batch_train.indptr
            ).max()
        )

        search_k = min(
            train_matrix.shape[1],
            CANDIDATE_K + max_seen,
        )

        start = time.perf_counter()

        batch_scores, batch_items = index.search(
            batch_vectors,
            search_k,
        )

        retrieval_time += (
            time.perf_counter()
            - start
        )

        for row, user_idx in enumerate(
            batch_users
        ):
            start = time.perf_counter()

            candidates, _ = get_unseen_candidates(
                batch_items[row],
                batch_scores[row],
                batch_train[row].indices,
            )

            _, _, order = rerank(
                candidates,
                batch_vectors[row],
                item_factors,
                popularity,
            )

            recommendations = candidates[
                order
            ]

            reranking_time += (
                time.perf_counter()
                - start
            )

            relevant = dev_matrix[
                user_idx
            ].indices

            # Candidate Recall@100 before reranking.
            candidate_recall_sum += (
                np.isin(
                    candidates,
                    relevant,
                ).sum()
                / len(relevant)
            )

            # Evaluate the same reranked list at each K.
            for k in K_VALUES:
                metric_sums[k] += evaluate_user(
                    recommendations,
                    relevant,
                    k,
                )

        completed = min(
            start_idx + BATCH_SIZE,
            len(users),
        )

        if completed % 10_000 < BATCH_SIZE:
            print(
                f"Evaluated: "
                f"{completed}/{len(users)}"
            )

    total_time = (
        time.perf_counter()
        - total_start
    )

    candidate_recall = (
        candidate_recall_sum
        / len(users)
    )

    # --------------------------------------------------------
    # Build and save TwoStage evaluation at all K values
    # --------------------------------------------------------

    evaluation_rows = []

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

        evaluation_rows.append(
            {
                "Model": "TwoStageHeuristic",
                "K": k,
                "EvaluatedUsers": len(users),
                "Recall": recall,
                "NDCG": ndcg,
                "MRR": mrr,
                "MAP": map_k,
                "HitRate": hit_rate,
                "CandidateRecall@100": candidate_recall,
                "ALSWeight": ALS_WEIGHT,
                "PopularityWeight": POPULARITY_WEIGHT,
            }
        )

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    evaluation.to_csv(
        DATA_DIR
        / "two_stage_evaluation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Direct three-model comparison at all K values
    # --------------------------------------------------------

    comparison = build_model_comparison(
        two_stage_evaluation=evaluation,
        evaluated_users=len(users),
    )

    comparison.to_csv(
        DATA_DIR
        / "two_stage_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save latency
    # --------------------------------------------------------

    latency = {
        "faiss_index_build_seconds": (
            index_build_time
        ),
        "retrieval_seconds": (
            retrieval_time
        ),
        "retrieval_ms_per_user": (
            1000
            * retrieval_time
            / len(users)
        ),
        "filter_and_rerank_seconds": (
            reranking_time
        ),
        "filter_and_rerank_ms_per_user": (
            1000
            * reranking_time
            / len(users)
        ),
        "end_to_end_seconds": (
            total_time
        ),
        "end_to_end_ms_per_user": (
            1000
            * total_time
            / len(users)
        ),
    }

    with open(
        DATA_DIR
        / "two_stage_latency.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            latency,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "FAISS index build time:",
        round(
            index_build_time,
            4,
        ),
    )

    print(
        "Maximum FAISS score error:",
        max_score_error,
    )

    print(
        "\nSample final top-10:"
    )

    print(
        final_table.to_string(
            index=False
        )
    )

    print(
        "\nTwoStage evaluation:"
    )

    print(
        evaluation.round(
            6
        ).to_string(
            index=False
        )
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
        "\nLatency:"
    )

    print(
        json.dumps(
            latency,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()