# ============================================================
# 15 - Full MIND Popularity and ALS Sample Recommendations
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"
NEWS_PATH = PROJECT_ROOT / "data" / "raw" / "MINDlarge_train" / "news.tsv"

TOP_K = 10


def build_result_table(
    user_idx,
    user_id,
    item_indices,
    scores,
    score_name,
    idx_item_map,
    title_map,
):
    """Convert item indices into readable recommendation results."""

    rows = []

    for rank, (item_idx, score) in enumerate(
        zip(item_indices, scores),
        start=1,
    ):
        item_idx = int(item_idx)
        news_id = idx_item_map[str(item_idx)]

        rows.append(
            {
                "user_idx": user_idx,
                "user_id": user_id,
                "rank": rank,
                "item_idx": item_idx,
                "news_id": news_id,
                "title": title_map.get(news_id, "Title not found"),
                score_name: float(score),
            }
        )

    return pd.DataFrame(rows)


def main():

    # Load Full MIND train matrix and ALS factors.
    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr()

    user_factors = np.load(
        DATA_DIR / "als_user_factors.npy",
        mmap_mode="r",
    )

    item_factors = np.load(
        DATA_DIR / "als_item_factors.npy",
        mmap_mode="r",
    )

    # Load reverse ID mappings.
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

    # Load news IDs and titles only.
    news = pd.read_csv(
        NEWS_PATH,
        sep="\t",
        header=None,
        usecols=[0, 3],
        names=["news_id", "title"],
    )

    title_map = dict(
        zip(news["news_id"], news["title"])
    )

    # Basic shape checks.
    if user_factors.shape[0] != train_matrix.shape[0]:
        raise ValueError("User-factor shape does not match train matrix.")

    if item_factors.shape[0] != train_matrix.shape[1]:
        raise ValueError("Item-factor shape does not match train matrix.")

    # --------------------------------------------------------
    # Item popularity
    # --------------------------------------------------------

    # The binary matrix column sum equals unique users per item.
    popularity = np.asarray(
        train_matrix.sum(axis=0)
    ).ravel()

    popularity_ranking = np.argsort(
        -popularity,
        kind="stable",
    )

    np.save(
        DATA_DIR / "popularity_scores.npy",
        popularity,
    )

    np.save(
        DATA_DIR / "popularity_ranking.npy",
        popularity_ranking,
    )

    # Select the first user with at least one train click.
    clicks_per_user = np.diff(train_matrix.indptr)
    user_idx = int(np.flatnonzero(clicks_per_user > 0)[0])
    user_id = idx_user_map[str(user_idx)]

    seen_items = train_matrix[user_idx].indices

    # --------------------------------------------------------
    # Popularity top-10 unseen items
    # --------------------------------------------------------

    popularity_items = popularity_ranking[
        ~np.isin(popularity_ranking, seen_items)
    ][:TOP_K]

    popularity_scores = popularity[
        popularity_items
    ]

    # --------------------------------------------------------
    # ALS top-10 unseen items
    # --------------------------------------------------------

    user_vector = np.asarray(
        user_factors[user_idx],
        dtype=np.float32,
    )

    als_scores = np.asarray(
        item_factors @ user_vector
    )

    # Exclude items already clicked in train.
    als_scores[seen_items] = -np.inf

    als_items = np.argsort(
        -als_scores,
        kind="stable",
    )[:TOP_K]

    top_als_scores = als_scores[
        als_items
    ]

    # Verify that no train-seen items are recommended.
    if np.intersect1d(popularity_items, seen_items).size:
        raise ValueError("Popularity contains train-seen items.")

    if np.intersect1d(als_items, seen_items).size:
        raise ValueError("ALS contains train-seen items.")

    # Verify ALS scores using direct dot products.
    if not np.allclose(
        top_als_scores,
        item_factors[als_items] @ user_vector,
        atol=1e-5,
    ):
        raise ValueError("ALS scores do not match factor dot products.")

    # Build readable outputs.
    popularity_result = build_result_table(
        user_idx,
        user_id,
        popularity_items,
        popularity_scores,
        "popularity_score",
        idx_item_map,
        title_map,
    )

    als_result = build_result_table(
        user_idx,
        user_id,
        als_items,
        top_als_scores,
        "als_score",
        idx_item_map,
        title_map,
    )

    # Save sample recommendations.
    popularity_result.to_csv(
        DATA_DIR / "popularity_sample_top10.csv",
        index=False,
    )

    als_result.to_csv(
        DATA_DIR / "als_sample_top10.csv",
        index=False,
    )

    print("Train matrix:", train_matrix.shape)
    print("Sample user:", user_idx, user_id)
    print("Train clicks:", len(seen_items))

    print("\nPopularity top-10:")
    print(popularity_result.to_string(index=False))

    print("\nALS top-10:")
    print(als_result.to_string(index=False))


if __name__ == "__main__":
    main()