# ============================================================
# 20 - Debiased Evaluation on MIND-small
#
# Inputs:
#   train_interactions.npz
#   dev_interactions.npz
#   six_model_top80_recommendations.npz
#   MINDsmall_train/behaviors.tsv
#
# Outputs:
#   item_propensity.csv
#   item_propensity.npy
#   item_inverse_propensity.npy
#   debiased_evaluation_comparison.csv
#   popularity_bias_comparison.csv
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

BEHAVIORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MINDsmall_train"
    / "behaviors.tsv"
)

MODELS = [
    "Popularity",
    "ItemKNN",
    "TruncatedSVD",
    "ALS",
    "BPR",
    "LightFM",
]

K_VALUES = [10, 20, 40, 80]

SMOOTHING = 1.0
MIN_PROPENSITY = 1e-4
MAX_INVERSE_WEIGHT = 100.0
HEAD_CLICK_SHARE = 0.80


# ------------------------------------------------------------
# NDCG values for one user
# ------------------------------------------------------------

def ndcg_values(
    recommended,
    relevant,
    inverse_weight,
    k,
):
    """Return naive NDCG, IPS NDCG, weighted DCG, weighted IDCG."""

    top_items = recommended[:k]

    hits = np.isin(
        top_items,
        relevant,
    ).astype(float)

    discounts = 1.0 / np.log2(
        np.arange(2, k + 2)
    )

    ideal_length = min(
        k,
        len(relevant),
    )

    # Ordinary NDCG.
    naive_dcg = np.sum(
        hits * discounts
    )

    naive_idcg = np.sum(
        discounts[:ideal_length]
    )

    naive_ndcg = (
        naive_dcg / naive_idcg
    )

    # IPS-weighted NDCG.
    weighted_dcg = np.sum(
        hits
        * inverse_weight[top_items]
        * discounts
    )

    ideal_weights = np.sort(
        inverse_weight[relevant]
    )[::-1][:ideal_length]

    weighted_idcg = np.sum(
        ideal_weights
        * discounts[:ideal_length]
    )

    ips_ndcg = (
        weighted_dcg / weighted_idcg
    )

    return (
        naive_ndcg,
        ips_ndcg,
        weighted_dcg,
        weighted_idcg,
    )


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

with open(
    DATA_DIR / "item_idx_map.json",
    encoding="utf-8",
) as file:
    item_idx_map = json.load(file)

with open(
    DATA_DIR / "idx_item_map.json",
    encoding="utf-8",
) as file:
    idx_item_map = json.load(file)

train_matrix = load_npz(
    DATA_DIR / "train_interactions.npz"
).tocsr()

dev_matrix = load_npz(
    DATA_DIR / "dev_interactions.npz"
).tocsr()

saved = np.load(
    DATA_DIR / "six_model_top80_recommendations.npz"
)

users = saved["evaluated_users"]
num_items = train_matrix.shape[1]

print("Evaluated users:", len(users))


# ------------------------------------------------------------
# Count item exposures from train impression logs
# ------------------------------------------------------------

exposure_counts = np.zeros(
    num_items,
    dtype=np.int64,
)

num_impressions = 0

with open(
    BEHAVIORS_PATH,
    encoding="utf-8",
) as file:

    for line in file:

        fields = line.rstrip().split(
            "\t",
            4,
        )

        if len(fields) != 5:
            continue

        num_impressions += 1

        # Each token looks like N12345-1 or N12345-0.
        for token in fields[4].split():

            news_id = token.rsplit(
                "-",
                1,
            )[0]

            item_idx = item_idx_map.get(
                news_id
            )

            if item_idx is not None:
                exposure_counts[
                    item_idx
                ] += 1


# ------------------------------------------------------------
# Propensity smoothing and clipping
# ------------------------------------------------------------

raw_propensity = (
    exposure_counts
    / num_impressions
)

smoothed_propensity = (
    exposure_counts
    + SMOOTHING
) / (
    num_impressions
    + 2 * SMOOTHING
)

clipped_propensity = np.clip(
    smoothed_propensity,
    MIN_PROPENSITY,
    1.0,
)

inverse_weight = np.minimum(
    1.0 / clipped_propensity,
    MAX_INVERSE_WEIGHT,
)

print(
    "Train impression rows:",
    num_impressions,
)

print(
    "Items with at least one exposure:",
    np.count_nonzero(
        exposure_counts
    ),
)

print(
    "Items at maximum inverse-weight clip:",
    np.count_nonzero(
        inverse_weight
        == MAX_INVERSE_WEIGHT
    ),
)


# ------------------------------------------------------------
# Save propensity results
# ------------------------------------------------------------

propensity_table = pd.DataFrame(
    {
        "item_idx": np.arange(
            num_items
        ),
        "news_id": [
            idx_item_map[str(i)]
            for i in range(num_items)
        ],
        "exposure_count": exposure_counts,
        "raw_propensity": raw_propensity,
        "smoothed_propensity": smoothed_propensity,
        "clipped_propensity": clipped_propensity,
        "inverse_weight": inverse_weight,
    }
)

propensity_table.to_csv(
    DATA_DIR / "item_propensity.csv",
    index=False,
)

np.save(
    DATA_DIR / "item_propensity.npy",
    clipped_propensity,
)

np.save(
    DATA_DIR / "item_inverse_propensity.npy",
    inverse_weight,
)


# ------------------------------------------------------------
# Define popularity head and long tail
# ------------------------------------------------------------

popularity = np.asarray(
    train_matrix.sum(axis=0)
).ravel()

popularity_order = np.argsort(
    -popularity,
    kind="stable",
)

cumulative_clicks = np.cumsum(
    popularity[
        popularity_order
    ]
)

head_count = (
    np.searchsorted(
        cumulative_clicks,
        HEAD_CLICK_SHARE
        * cumulative_clicks[-1],
    )
    + 1
)

head_mask = np.zeros(
    num_items,
    dtype=bool,
)

head_mask[
    popularity_order[:head_count]
] = True

# Long-tail items have train clicks but are outside the head.
long_tail_mask = (
    (~head_mask)
    & (popularity > 0)
)

print(
    "Head items covering 80% of clicks:",
    head_count,
)

print(
    "Long-tail items with train clicks:",
    long_tail_mask.sum(),
)


# ------------------------------------------------------------
# Evaluate all models
# ------------------------------------------------------------

debiased_rows = []
popularity_bias_rows = []

for model_name in MODELS:

    print(
        "Evaluating:",
        model_name,
    )

    recommendations = saved[
        model_name
    ]

    for k in K_VALUES:

        naive_sum = 0.0
        ips_sum = 0.0
        weighted_dcg_sum = 0.0
        weighted_idcg_sum = 0.0

        for row, user_idx in enumerate(
            users
        ):

            relevant = dev_matrix[
                user_idx
            ].indices

            (
                naive_ndcg,
                ips_ndcg,
                weighted_dcg,
                weighted_idcg,
            ) = ndcg_values(
                recommendations[row],
                relevant,
                inverse_weight,
                k,
            )

            naive_sum += naive_ndcg
            ips_sum += ips_ndcg
            weighted_dcg_sum += weighted_dcg
            weighted_idcg_sum += weighted_idcg

        # SNIPS uses the ratio of aggregate weighted gains.
        snips_ndcg = (
            weighted_dcg_sum
            / weighted_idcg_sum
        )

        debiased_rows.append(
            {
                "Model": model_name,
                "K": k,
                "EvaluatedUsers": len(users),
                "NaiveNDCG": (
                    naive_sum
                    / len(users)
                ),
                "IPSNDCG": (
                    ips_sum
                    / len(users)
                ),
                "SNIPSNDCG": snips_ndcg,
                "DebiasedNDCG": snips_ndcg,
            }
        )

        top_items = recommendations[
            :,
            :k,
        ]

        popularity_bias_rows.append(
            {
                "Model": model_name,
                "K": k,
                "AverageRecommendationPopularity": float(
                    popularity[
                        top_items
                    ].mean()
                ),
                "AverageLogRecommendationPopularity": float(
                    np.log1p(
                        popularity[
                            top_items
                        ]
                    ).mean()
                ),
                "LongTailRecommendationShare": float(
                    long_tail_mask[
                        top_items
                    ].mean()
                ),
                "CatalogCoverage": (
                    len(
                        np.unique(
                            top_items
                        )
                    )
                    / num_items
                ),
            }
        )


# ------------------------------------------------------------
# Sort and save result tables
# ------------------------------------------------------------

model_order = {
    model: index
    for index, model
    in enumerate(MODELS)
}

debiased_results = pd.DataFrame(
    debiased_rows
)

debiased_results["_order"] = (
    debiased_results["Model"]
    .map(model_order)
)

debiased_results = (
    debiased_results
    .sort_values(
        ["K", "_order"]
    )
    .drop(columns="_order")
    .reset_index(drop=True)
)

popularity_bias_results = pd.DataFrame(
    popularity_bias_rows
)

popularity_bias_results["_order"] = (
    popularity_bias_results["Model"]
    .map(model_order)
)

popularity_bias_results = (
    popularity_bias_results
    .sort_values(
        ["K", "_order"]
    )
    .drop(columns="_order")
    .reset_index(drop=True)
)

debiased_results.to_csv(
    DATA_DIR
    / "debiased_evaluation_comparison.csv",
    index=False,
)

popularity_bias_results.to_csv(
    DATA_DIR
    / "popularity_bias_comparison.csv",
    index=False,
)


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\nDebiased evaluation:")

print(
    debiased_results
    .round(6)
    .to_string(index=False)
)

print("\nPopularity-bias evaluation:")

print(
    popularity_bias_results
    .round(6)
    .to_string(index=False)
)

print(
    "\nSaved:",
    DATA_DIR
    / "item_propensity.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "debiased_evaluation_comparison.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "popularity_bias_comparison.csv",
)