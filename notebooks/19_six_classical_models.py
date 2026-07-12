# ============================================================
# 19 - Six Classical Models on MIND-small
#
# Models:
# Popularity, ItemKNN, Truncated SVD, ALS, BPR, LightFM-Hybrid
#
# Outputs:
# data/processed/six_model_comparison.csv
# data/processed/six_model_top80_recommendations.npz
# ============================================================

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, load_npz
from sklearn.decomposition import TruncatedSVD

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from lightfm import LightFM


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

K_VALUES = [10, 20, 40, 80]
MAX_K = max(K_VALUES)

SEED = 42
THREADS = min(8, os.cpu_count() or 1)

# Whole-catalog scoring creates a dense batch x items score matrix.
DENSE_BATCH_SIZE = 128
IMPLICIT_BATCH_SIZE = 512

MODEL_NAMES = [
    "Popularity",
    "ItemKNN",
    "TruncatedSVD",
    "ALS",
    "BPR",
    "LightFM",
]


# ------------------------------------------------------------
# Ranking metrics
# ------------------------------------------------------------

def evaluate_user(recommended, relevant, k):
    """Recall, NDCG, MRR, MAP, and Hit Rate for one user."""

    hits = np.isin(
        recommended[:k],
        relevant,
    ).astype(float)

    recall = hits.sum() / len(relevant)

    discounts = 1.0 / np.log2(
        np.arange(2, len(hits) + 2)
    )

    dcg = np.sum(hits * discounts)

    ideal_length = min(k, len(relevant))

    idcg = np.sum(
        1.0 / np.log2(
            np.arange(2, ideal_length + 2)
        )
    )

    ndcg = dcg / idcg

    hit_positions = np.flatnonzero(hits)

    mrr = (
        0.0
        if len(hit_positions) == 0
        else 1.0 / (hit_positions[0] + 1)
    )

    precision_at_rank = (
        np.cumsum(hits)
        / np.arange(1, len(hits) + 1)
    )

    map_k = (
        np.sum(precision_at_rank * hits)
        / ideal_length
    )

    hit_rate = float(hits.sum() > 0)

    return np.array(
        [recall, ndcg, mrr, map_k, hit_rate]
    )


def evaluate_model(
    model_name,
    recommendations,
    users,
    dev_matrix,
    training_time,
    recommendation_time,
):
    """Evaluate one saved top-80 recommendation matrix."""

    sums = {
        k: np.zeros(5)
        for k in K_VALUES
    }

    for row, user_idx in enumerate(users):

        relevant = dev_matrix[user_idx].indices

        for k in K_VALUES:
            sums[k] += evaluate_user(
                recommendations[row],
                relevant,
                k,
            )

    rows = []

    for k in K_VALUES:

        recall, ndcg, mrr, map_k, hit_rate = (
            sums[k] / len(users)
        )

        rows.append(
            {
                "Model": model_name,
                "K": k,
                "EvaluatedUsers": len(users),
                "Recall": recall,
                "NDCG": ndcg,
                "MRR": mrr,
                "MAP": map_k,
                "HitRate": hit_rate,
                "TrainingTimeSeconds": training_time,
                "RecommendationTimeSeconds": recommendation_time,
                "RecommendationMillisecondsPerUser": (
                    recommendation_time
                    * 1000
                    / len(users)
                ),
            }
        )

    return rows


# ------------------------------------------------------------
# Recommendation helpers
# ------------------------------------------------------------

def fill_with_popularity(
    raw_items,
    user_idx,
    train_matrix,
    popularity_ranking,
):
    """
    Keep valid model recommendations, then use Popularity to fill
    missing positions.

    implicit ItemKNN may return -1 when it cannot produce 80 items.
    """

    seen = set(
        train_matrix[user_idx].indices
    )

    selected = []
    selected_set = set()

    for item_idx in raw_items:

        item_idx = int(item_idx)

        if item_idx < 0:
            continue

        if item_idx in seen or item_idx in selected_set:
            continue

        selected.append(item_idx)
        selected_set.add(item_idx)

        if len(selected) == MAX_K:
            break

    # Fill missing positions with popular unseen items.
    if len(selected) < MAX_K:

        for item_idx in popularity_ranking:

            item_idx = int(item_idx)

            if item_idx in seen or item_idx in selected_set:
                continue

            selected.append(item_idx)
            selected_set.add(item_idx)

            if len(selected) == MAX_K:
                break

    if len(selected) != MAX_K:
        raise ValueError(
            f"Could not generate {MAX_K} items "
            f"for user {user_idx}."
        )

    return np.asarray(
        selected,
        dtype=np.int32,
    )


def recommend_popularity(
    users,
    train_matrix,
    popularity_ranking,
):
    """Recommend globally popular unseen items."""

    results = np.empty(
        (len(users), MAX_K),
        dtype=np.int32,
    )

    for row, user_idx in enumerate(users):

        results[row] = fill_with_popularity(
            raw_items=[],
            user_idx=user_idx,
            train_matrix=train_matrix,
            popularity_ranking=popularity_ranking,
        )

    return results


def recommend_implicit(
    model,
    users,
    train_matrix,
    popularity_ranking,
):
    """
    Batch recommendation for ItemKNN, ALS, and BPR.

    Invalid -1 values are removed and missing positions are filled
    with popular unseen items.
    """

    parts = []
    fallback_users = 0

    for start in range(
        0,
        len(users),
        IMPLICIT_BATCH_SIZE,
    ):
        batch_users = users[
            start:start + IMPLICIT_BATCH_SIZE
        ]

        raw_items, _ = model.recommend(
            userid=batch_users,
            user_items=train_matrix[batch_users],
            N=MAX_K,
            filter_already_liked_items=True,
        )

        raw_items = np.atleast_2d(raw_items)

        batch_results = np.empty(
            (len(batch_users), MAX_K),
            dtype=np.int32,
        )

        for row, user_idx in enumerate(batch_users):

            if np.any(raw_items[row] < 0):
                fallback_users += 1

            batch_results[row] = fill_with_popularity(
                raw_items=raw_items[row],
                user_idx=user_idx,
                train_matrix=train_matrix,
                popularity_ranking=popularity_ranking,
            )

        parts.append(batch_results)

        print(
            f"  recommended "
            f"{min(start + IMPLICIT_BATCH_SIZE, len(users))}"
            f"/{len(users)} users"
        )

    print(
        "  users requiring fallback:",
        fallback_users,
    )

    return np.vstack(parts)


def dense_top_k(scores, train_rows):
    """Filter train-seen items and extract top-80 from dense scores."""

    for row in range(scores.shape[0]):

        scores[
            row,
            train_rows[row].indices,
        ] = -np.inf

    # Select only the largest 80 scores before sorting them.
    candidates = np.argpartition(
        scores,
        kth=scores.shape[1] - MAX_K,
        axis=1,
    )[:, -MAX_K:]

    candidate_scores = np.take_along_axis(
        scores,
        candidates,
        axis=1,
    )

    order = np.argsort(
        -candidate_scores,
        axis=1,
    )

    return np.take_along_axis(
        candidates,
        order,
        axis=1,
    ).astype(np.int32)


def recommend_from_factors(
    user_factors,
    item_factors,
    users,
    train_matrix,
    item_biases=None,
):
    """
    Whole-catalog scoring for Truncated SVD and LightFM.

    score = user factor dot item factor
    LightFM additionally uses item biases.
    """

    parts = []

    for start in range(
        0,
        len(users),
        DENSE_BATCH_SIZE,
    ):
        batch_users = users[
            start:start + DENSE_BATCH_SIZE
        ]

        scores = (
            np.asarray(
                user_factors[batch_users],
                dtype=np.float32,
            )
            @ np.asarray(
                item_factors,
                dtype=np.float32,
            ).T
        )

        if item_biases is not None:
            scores += item_biases[None, :]

        parts.append(
            dense_top_k(
                scores,
                train_matrix[batch_users],
            )
        )

        print(
            f"  recommended "
            f"{min(start + DENSE_BATCH_SIZE, len(users))}"
            f"/{len(users)} users"
        )

    return np.vstack(parts)


def validate_recommendations(
    model_name,
    recommendations,
    users,
    train_matrix,
):
    """Check shape, valid indices, duplicates, and train-seen items."""

    if recommendations.shape != (
        len(users),
        MAX_K,
    ):
        raise ValueError(
            f"{model_name}: wrong recommendation shape."
        )

    if (
        recommendations.min() < 0
        or recommendations.max() >= train_matrix.shape[1]
    ):
        raise ValueError(
            f"{model_name}: invalid item index."
        )

    for row, user_idx in enumerate(users):

        items = recommendations[row]

        if len(np.unique(items)) != MAX_K:
            raise ValueError(
                f"{model_name}: duplicate items."
            )

        if np.intersect1d(
            items,
            train_matrix[user_idx].indices,
        ).size:
            raise ValueError(
                f"{model_name}: train-seen item found."
            )

    print(
        f"{model_name}: validation passed."
    )


# ------------------------------------------------------------
# LightFM item features
# ------------------------------------------------------------

def build_lightfm_features(
    num_items,
    idx_item_map,
    news,
):
    """
    Each item receives three features:

    1. item identity
    2. category
    3. subcategory
    """

    items = pd.DataFrame(
        {
            "item_idx": np.arange(num_items),
            "news_id": [
                idx_item_map[str(i)]
                for i in range(num_items)
            ],
        }
    )

    metadata = news[
        ["news_id", "category", "subcategory"]
    ].drop_duplicates("news_id")

    items = items.merge(
        metadata,
        on="news_id",
        how="left",
    )

    category_codes = pd.factorize(
        items["category"]
        .fillna("__MISSING__")
        .astype(str)
    )[0]

    subcategory_codes = pd.factorize(
        items["subcategory"]
        .fillna("__MISSING__")
        .astype(str)
    )[0]

    num_categories = category_codes.max() + 1
    num_subcategories = subcategory_codes.max() + 1

    item_rows = np.arange(
        num_items,
        dtype=np.int32,
    )

    rows = np.concatenate(
        [item_rows, item_rows, item_rows]
    )

    columns = np.concatenate(
        [
            # Identity features.
            item_rows,

            # Category features.
            num_items + category_codes,

            # Subcategory features.
            (
                num_items
                + num_categories
                + subcategory_codes
            ),
        ]
    )

    return coo_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (rows, columns),
        ),
        shape=(
            num_items,
            (
                num_items
                + num_categories
                + num_subcategories
            ),
        ),
        dtype=np.float32,
    ).tocsr()


# ------------------------------------------------------------
# Train one model, generate recommendations, and record time
# ------------------------------------------------------------

def save_result(
    model_name,
    model_recommendations,
    users,
    train_matrix,
    train_time,
    recommend_time,
    recommendations,
    training_times,
    recommendation_times,
):
    validate_recommendations(
        model_name,
        model_recommendations,
        users,
        train_matrix,
    )

    recommendations[model_name] = (
        model_recommendations
    )

    training_times[model_name] = train_time
    recommendation_times[model_name] = (
        recommend_time
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # Simple metric sanity checks.
    assert np.allclose(
        evaluate_user([1, 2], [1, 2], 2),
        1.0,
    )

    assert np.allclose(
        evaluate_user([3, 4], [1, 2], 2),
        0.0,
    )

    print("Metric sanity checks passed.")

    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr().astype(np.float32)

    dev_matrix = load_npz(
        DATA_DIR / "dev_interactions.npz"
    ).tocsr().astype(np.float32)

    if train_matrix.shape != dev_matrix.shape:
        raise ValueError(
            "Train and dev matrix shapes do not match."
        )

    with open(
        DATA_DIR / "idx_item_map.json",
        encoding="utf-8",
    ) as file:
        idx_item_map = json.load(file)

    news = pd.read_parquet(
        DATA_DIR / "news.parquet",
        columns=[
            "news_id",
            "category",
            "subcategory",
        ],
    )

    # Same warm-start evaluated users as the earlier evaluation.
    users = np.flatnonzero(
        np.diff(dev_matrix.indptr) > 0
    )

    print("Train matrix:", train_matrix.shape)
    print("Train nnz:", train_matrix.nnz)
    print("Dev nnz:", dev_matrix.nnz)
    print("Evaluated users:", len(users))

    recommendations = {}
    training_times = {}
    recommendation_times = {}

    # ========================================================
    # 1. Popularity
    # ========================================================

    print("\n=== Popularity ===")

    start = time.perf_counter()

    popularity = np.asarray(
        train_matrix.sum(axis=0)
    ).ravel()

    popularity_ranking = np.argsort(
        -popularity,
        kind="stable",
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    model_recommendations = recommend_popularity(
        users,
        train_matrix,
        popularity_ranking,
    )

    recommend_time = time.perf_counter() - start

    save_result(
        "Popularity",
        model_recommendations,
        users,
        train_matrix,
        train_time,
        recommend_time,
        recommendations,
        training_times,
        recommendation_times,
    )

    # ========================================================
    # 2. ItemKNN
    # ========================================================

    print("\n=== ItemKNN ===")

    model = CosineRecommender(
        K=100,
        num_threads=THREADS,
    )

    start = time.perf_counter()

    model.fit(
        train_matrix,
        show_progress=True,
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    model_recommendations = recommend_implicit(
        model,
        users,
        train_matrix,
        popularity_ranking,
    )

    recommend_time = time.perf_counter() - start

    save_result(
        "ItemKNN",
        model_recommendations,
        users,
        train_matrix,
        train_time,
        recommend_time,
        recommendations,
        training_times,
        recommendation_times,
    )

    # ========================================================
    # 3. Truncated SVD
    # ========================================================

    print("\n=== Truncated SVD ===")

    model = TruncatedSVD(
        n_components=64,
        n_iter=7,
        random_state=SEED,
    )

    start = time.perf_counter()

    # fit_transform returns U Sigma.
    user_factors = model.fit_transform(
        train_matrix
    ).astype(np.float32)

    # components_.T contains one latent vector per item.
    item_factors = model.components_.T.astype(
        np.float32
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    model_recommendations = recommend_from_factors(
        user_factors,
        item_factors,
        users,
        train_matrix,
    )

    recommend_time = time.perf_counter() - start

    save_result(
        "TruncatedSVD",
        model_recommendations,
        users,
        train_matrix,
        train_time,
        recommend_time,
        recommendations,
        training_times,
        recommendation_times,
    )

    # Free memory before training the next model.
    del user_factors, item_factors, model

    # ========================================================
    # 4. ALS
    # ========================================================

    print("\n=== ALS ===")

    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.1,
        alpha=40.0,
        iterations=15,
        num_threads=THREADS,
        random_state=SEED,
    )

    start = time.perf_counter()

    model.fit(
        train_matrix,
        show_progress=True,
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    model_recommendations = recommend_implicit(
        model,
        users,
        train_matrix,
        popularity_ranking,
    )

    recommend_time = time.perf_counter() - start

    save_result(
        "ALS",
        model_recommendations,
        users,
        train_matrix,
        train_time,
        recommend_time,
        recommendations,
        training_times,
        recommendation_times,
    )

    del model

    # ========================================================
    # 5. BPR
    # ========================================================

    print("\n=== BPR ===")

    model = BayesianPersonalizedRanking(
        factors=64,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
        num_threads=THREADS,
        verify_negative_samples=True,
        random_state=SEED,
    )

    start = time.perf_counter()

    model.fit(
        train_matrix,
        show_progress=True,
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    model_recommendations = recommend_implicit(
        model,
        users,
        train_matrix,
        popularity_ranking,
    )

    recommend_time = time.perf_counter() - start

    save_result(
        "BPR",
        model_recommendations,
        users,
        train_matrix,
        train_time,
        recommend_time,
        recommendations,
        training_times,
        recommendation_times,
    )

    del model

    # ========================================================
    # 6. LightFM-Hybrid
    # ========================================================

    print("\n=== LightFM-Hybrid ===")

    start = time.perf_counter()

    item_features = build_lightfm_features(
        train_matrix.shape[1],
        idx_item_map,
        news,
    )

    model = LightFM(
        no_components=64,
        loss="warp",
        learning_rate=0.05,
        user_alpha=1e-6,
        item_alpha=1e-6,
        random_state=SEED,
    )

    model.fit(
        train_matrix,
        item_features=item_features,
        epochs=20,
        num_threads=THREADS,
        verbose=True,
    )

    # Final representations already include feature embeddings.
    item_biases, item_factors = (
        model.get_item_representations(
            item_features
        )
    )

    _, user_factors = (
        model.get_user_representations()
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    model_recommendations = recommend_from_factors(
        user_factors,
        item_factors,
        users,
        train_matrix,
        item_biases=item_biases,
    )

    recommend_time = time.perf_counter() - start

    save_result(
        "LightFM",
        model_recommendations,
        users,
        train_matrix,
        train_time,
        recommend_time,
        recommendations,
        training_times,
        recommendation_times,
    )

    # --------------------------------------------------------
    # Save top-80 recommendations for the debiased evaluation.
    # --------------------------------------------------------

    np.savez_compressed(
        DATA_DIR
        / "six_model_top80_recommendations.npz",
        evaluated_users=users,
        **recommendations,
    )

    # --------------------------------------------------------
    # Evaluate all six models.
    # --------------------------------------------------------

    rows = []

    for model_name in MODEL_NAMES:

        print(
            f"Evaluating {model_name}..."
        )

        rows.extend(
            evaluate_model(
                model_name,
                recommendations[model_name],
                users,
                dev_matrix,
                training_times[model_name],
                recommendation_times[model_name],
            )
        )

    comparison = pd.DataFrame(rows)

    order = {
        name: position
        for position, name
        in enumerate(MODEL_NAMES)
    }

    comparison["_order"] = (
        comparison["Model"].map(order)
    )

    comparison = (
        comparison
        .sort_values(["K", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    comparison.to_csv(
        DATA_DIR
        / "six_model_comparison.csv",
        index=False,
    )

    print("\nSix-model comparison:")

    print(
        comparison
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nSaved:",
        DATA_DIR
        / "six_model_comparison.csv",
    )

    print(
        "Saved:",
        DATA_DIR
        / "six_model_top80_recommendations.npz",
    )


if __name__ == "__main__":
    main()