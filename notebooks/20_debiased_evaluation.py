# ============================================================
# 20 - Debiased Evaluation on MIND-small
#
# Reuses the six-model top-80 recommendations from Step 19.
# No recommendation model is retrained.
#
# Main additions:
# 1. Item-level marginal exposure propensity proxy.
# 2. Weight-clipping sensitivity: 100, 500, 1000.
# 3. Zero-exposure support diagnostics.
# 4. Naive, IPS-weighted, and SNIPS-style NDCG.
# 5. Popularity-bias metrics.
# 6. Paired user-level bootstrap confidence intervals.
#
# Important:
# The propensity is not the true logging probability P(E_ui=1|u,i).
# This remains a counterfactual-style evaluation.
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

# Compare several clipping levels instead of relying on one value.
MAX_WEIGHTS = [100.0, 500.0, 1000.0]

# Use the middle threshold for the main table and bootstrap analysis.
PRIMARY_MAX_WEIGHT = 500.0

HEAD_CLICK_SHARE = 0.80

BOOTSTRAP_K = 80
BOOTSTRAP_SAMPLES = 1000
SEED = 42


# ------------------------------------------------------------
# Return per-user NDCG components
# ------------------------------------------------------------

def user_values(
    recommendations,
    users,
    dev_matrix,
    inverse_weight,
    exposure_counts,
    k,
    supported_only=False,
):
    """
    Return one value per evaluated user:

    naive NDCG
    IPS-weighted NDCG
    weighted DCG
    weighted ideal DCG

    If supported_only=True, dev relevant items with zero observed
    train exposure are removed. Users with no remaining targets
    are excluded from that evaluation scope.
    """

    naive_values = []
    ips_values = []
    weighted_dcg_values = []
    weighted_idcg_values = []

    for row, user_idx in enumerate(users):

        relevant = dev_matrix[user_idx].indices

        if supported_only:
            relevant = relevant[
                exposure_counts[relevant] > 0
            ]

        if len(relevant) == 0:
            continue

        top_items = recommendations[row, :k]

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

        naive_values.append(
            naive_dcg / naive_idcg
        )

        # Propensity-weighted NDCG.
        weighted_dcg = np.sum(
            hits
            * inverse_weight[top_items]
            * discounts
        )

        # The ideal ranking places the largest relevant inverse
        # weights at the highest ranks.
        ideal_weights = np.sort(
            inverse_weight[relevant]
        )[::-1][:ideal_length]

        weighted_idcg = np.sum(
            ideal_weights
            * discounts[:ideal_length]
        )

        ips_values.append(
            weighted_dcg / weighted_idcg
        )

        weighted_dcg_values.append(
            weighted_dcg
        )

        weighted_idcg_values.append(
            weighted_idcg
        )

    return (
        np.asarray(naive_values),
        np.asarray(ips_values),
        np.asarray(weighted_dcg_values),
        np.asarray(weighted_idcg_values),
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
    DATA_DIR / "six_model_top80_recommendations.npz",
    allow_pickle=False,
)

users = saved["evaluated_users"]
num_items = train_matrix.shape[1]

print("Evaluated users:", len(users))


# ------------------------------------------------------------
# Count item exposures in train impression logs
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

        fields = line.rstrip(
            "\r\n"
        ).split(
            "\t",
            4,
        )

        if len(fields) != 5:
            continue

        num_impressions += 1

        # Each token is news_id-click_label.
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
# Propensity estimation
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

inverse_weights = {
    max_weight: np.minimum(
        1.0 / clipped_propensity,
        max_weight,
    )
    for max_weight in MAX_WEIGHTS
}

zero_exposure_mask = (
    exposure_counts == 0
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

for max_weight in MAX_WEIGHTS:

    clipped_count = np.count_nonzero(
        inverse_weights[max_weight]
        == max_weight
    )

    print(
        f"Items clipped at {max_weight:g}:",
        clipped_count,
        f"({clipped_count / num_items:.2%})",
    )


# ------------------------------------------------------------
# Save item propensity table
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
    }
)

for max_weight in MAX_WEIGHTS:
    propensity_table[
        f"inverse_weight_{int(max_weight)}"
    ] = inverse_weights[max_weight]

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
    inverse_weights[
        PRIMARY_MAX_WEIGHT
    ],
)


# ------------------------------------------------------------
# Global support diagnostics
# ------------------------------------------------------------

dev_positive_items = dev_matrix.indices

zero_exposure_dev_pairs = int(
    zero_exposure_mask[
        dev_positive_items
    ].sum()
)

unique_dev_items = np.unique(
    dev_positive_items
)

zero_exposure_unique_dev_items = int(
    zero_exposure_mask[
        unique_dev_items
    ].sum()
)

support_summary = {
    "train_known_items": int(
        num_items
    ),
    "items_with_positive_exposure": int(
        np.count_nonzero(
            exposure_counts
        )
    ),
    "items_with_zero_exposure": int(
        zero_exposure_mask.sum()
    ),
    "dev_positive_pairs": int(
        dev_matrix.nnz
    ),
    "zero_exposure_dev_positive_pairs": (
        zero_exposure_dev_pairs
    ),
    "unique_dev_positive_items": int(
        len(unique_dev_items)
    ),
    "zero_exposure_unique_dev_positive_items": (
        zero_exposure_unique_dev_items
    ),
}

with open(
    DATA_DIR / "propensity_support_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        support_summary,
        file,
        indent=2,
    )

print(
    "Zero-exposure dev positive pairs:",
    zero_exposure_dev_pairs,
)

print(
    "Zero-exposure unique dev positive items:",
    zero_exposure_unique_dev_items,
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

# Long-tail items have at least one train click but do not belong
# to the popularity head.
long_tail_mask = (
    (~head_mask)
    & (popularity > 0)
)

active_item_mask = (
    popularity > 0
)

print(
    "Head items covering 80% of clicks:",
    head_count,
)

print(
    "Long-tail items with train clicks:",
    int(long_tail_mask.sum()),
)


# ------------------------------------------------------------
# Weight-clipping sensitivity evaluation
# ------------------------------------------------------------

sensitivity_rows = []

for max_weight in MAX_WEIGHTS:

    inverse_weight = inverse_weights[
        max_weight
    ]

    clipped_count = np.count_nonzero(
        inverse_weight == max_weight
    )

    for scope_name, supported_only in [
        ("AllItems", False),
        ("SupportedRelevantOnly", True),
    ]:

        for model_name in MODELS:

            recommendations = saved[
                model_name
            ]

            print(
                "Evaluating:",
                model_name,
                "| scope:",
                scope_name,
                "| max weight:",
                max_weight,
            )

            for k in K_VALUES:

                (
                    naive_values,
                    ips_values,
                    weighted_dcg_values,
                    weighted_idcg_values,
                ) = user_values(
                    recommendations=(
                        recommendations
                    ),
                    users=users,
                    dev_matrix=dev_matrix,
                    inverse_weight=(
                        inverse_weight
                    ),
                    exposure_counts=(
                        exposure_counts
                    ),
                    k=k,
                    supported_only=(
                        supported_only
                    ),
                )

                # Ratio of aggregate weighted gains.
                # This is SNIPS-style, not a strict policy-value
                # SNIPS estimator.
                snips_style_ndcg = (
                    weighted_dcg_values.sum()
                    / weighted_idcg_values.sum()
                )

                sensitivity_rows.append(
                    {
                        "Model": model_name,
                        "K": k,
                        "Scope": scope_name,
                        "EvaluatedUsers": len(
                            naive_values
                        ),
                        "MaxInverseWeight": (
                            max_weight
                        ),
                        "ClippedItems": (
                            clipped_count
                        ),
                        "ClippedItemShare": (
                            clipped_count
                            / num_items
                        ),
                        "NaiveNDCG": float(
                            naive_values.mean()
                        ),
                        "IPSNDCG": float(
                            ips_values.mean()
                        ),
                        "SNIPSStyleNDCG": float(
                            snips_style_ndcg
                        ),
                    }
                )

sensitivity_results = pd.DataFrame(
    sensitivity_rows
)

model_order = {
    model: index
    for index, model
    in enumerate(MODELS)
}

sensitivity_results["_order"] = (
    sensitivity_results["Model"]
    .map(model_order)
)

sensitivity_results = (
    sensitivity_results
    .sort_values(
        [
            "MaxInverseWeight",
            "Scope",
            "K",
            "_order",
        ]
    )
    .drop(columns="_order")
    .reset_index(drop=True)
)

sensitivity_results.to_csv(
    DATA_DIR
    / "debiased_evaluation_sensitivity.csv",
    index=False,
)

# Main table uses the middle clipping threshold and all targets.
debiased_results = sensitivity_results[
    (
        sensitivity_results[
            "MaxInverseWeight"
        ]
        == PRIMARY_MAX_WEIGHT
    )
    & (
        sensitivity_results[
            "Scope"
        ]
        == "AllItems"
    )
].reset_index(drop=True)

debiased_results.to_csv(
    DATA_DIR
    / "debiased_evaluation_comparison.csv",
    index=False,
)


# ------------------------------------------------------------
# Popularity-bias and support diagnostics
# ------------------------------------------------------------

popularity_bias_rows = []
support_rows = []

for model_name in MODELS:

    recommendations = saved[
        model_name
    ]

    for k in K_VALUES:

        top_items = recommendations[
            :,
            :k,
        ]

        active_unique_items = np.unique(
            top_items[
                active_item_mask[
                    top_items
                ]
            ]
        )

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
                "ZeroPopularityRecommendationShare": float(
                    (
                        popularity[
                            top_items
                        ]
                        == 0
                    ).mean()
                ),
                "CatalogCoverage": float(
                    len(
                        np.unique(
                            top_items
                        )
                    )
                    / num_items
                ),
                "ActiveCatalogCoverage": float(
                    len(
                        active_unique_items
                    )
                    / active_item_mask.sum()
                ),
            }
        )

        total_hits = 0
        zero_exposure_hits = 0

        for row, user_idx in enumerate(
            users
        ):

            user_top_items = top_items[
                row
            ]

            hit_mask = np.isin(
                user_top_items,
                dev_matrix[
                    user_idx
                ].indices,
            )

            total_hits += int(
                hit_mask.sum()
            )

            zero_exposure_hits += int(
                (
                    hit_mask
                    & zero_exposure_mask[
                        user_top_items
                    ]
                ).sum()
            )

        support_rows.append(
            {
                "Model": model_name,
                "K": k,
                "ZeroExposureRecommendationShare": float(
                    zero_exposure_mask[
                        top_items
                    ].mean()
                ),
                "ZeroExposureRecommendedUniqueItems": int(
                    zero_exposure_mask[
                        np.unique(
                            top_items
                        )
                    ].sum()
                ),
                "TotalRecommendationHits": (
                    total_hits
                ),
                "ZeroExposureRecommendationHits": (
                    zero_exposure_hits
                ),
                "ZeroExposureHitShare": (
                    zero_exposure_hits
                    / total_hits
                    if total_hits > 0
                    else 0.0
                ),
            }
        )

popularity_bias_results = pd.DataFrame(
    popularity_bias_rows
)

support_results = pd.DataFrame(
    support_rows
)

for result in [
    popularity_bias_results,
    support_results,
]:
    result["_order"] = (
        result["Model"]
        .map(model_order)
    )

    result.sort_values(
        ["K", "_order"],
        inplace=True,
    )

    result.drop(
        columns="_order",
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

popularity_bias_results.to_csv(
    DATA_DIR
    / "popularity_bias_comparison.csv",
    index=False,
)

support_results.to_csv(
    DATA_DIR
    / "recommendation_support_diagnostics.csv",
    index=False,
)


# ------------------------------------------------------------
# Paired user-level bootstrap at K = 80
# ------------------------------------------------------------

primary_weight = inverse_weights[
    PRIMARY_MAX_WEIGHT
]

rng = np.random.default_rng(
    SEED
)

# The same sampled users are used for every model, making model
# comparisons paired.
bootstrap_indices = rng.integers(
    0,
    len(users),
    size=(
        BOOTSTRAP_SAMPLES,
        len(users),
    ),
    dtype=np.int32,
)

bootstrap_model_rows = []
bootstrap_samples = {}

for model_name in MODELS:

    (
        naive_values,
        _,
        weighted_dcg_values,
        weighted_idcg_values,
    ) = user_values(
        recommendations=saved[
            model_name
        ],
        users=users,
        dev_matrix=dev_matrix,
        inverse_weight=primary_weight,
        exposure_counts=exposure_counts,
        k=BOOTSTRAP_K,
        supported_only=False,
    )

    naive_bootstrap = (
        naive_values[
            bootstrap_indices
        ].mean(
            axis=1
        )
    )

    snips_bootstrap = (
        weighted_dcg_values[
            bootstrap_indices
        ].sum(
            axis=1
        )
        / weighted_idcg_values[
            bootstrap_indices
        ].sum(
            axis=1
        )
    )

    bootstrap_samples[
        model_name
    ] = {
        "NaiveNDCG": (
            naive_bootstrap
        ),
        "SNIPSStyleNDCG": (
            snips_bootstrap
        ),
    }

    for metric_name, samples in [
        (
            "NaiveNDCG",
            naive_bootstrap,
        ),
        (
            "SNIPSStyleNDCG",
            snips_bootstrap,
        ),
    ]:

        bootstrap_model_rows.append(
            {
                "Model": model_name,
                "K": BOOTSTRAP_K,
                "Metric": metric_name,
                "MaxInverseWeight": (
                    PRIMARY_MAX_WEIGHT
                ),
                "Estimate": float(
                    samples.mean()
                ),
                "CI95Lower": float(
                    np.quantile(
                        samples,
                        0.025,
                    )
                ),
                "CI95Upper": float(
                    np.quantile(
                        samples,
                        0.975,
                    )
                ),
            }
        )

bootstrap_results = pd.DataFrame(
    bootstrap_model_rows
)

bootstrap_results.to_csv(
    DATA_DIR
    / "bootstrap_confidence_intervals.csv",
    index=False,
)


# ------------------------------------------------------------
# Paired bootstrap model differences
# ------------------------------------------------------------

comparison_pairs = [
    ("LightFM", "ALS"),
    ("LightFM", "BPR"),
    ("BPR", "ALS"),
]

pairwise_rows = []

for model_a, model_b in comparison_pairs:

    for metric_name in [
        "NaiveNDCG",
        "SNIPSStyleNDCG",
    ]:

        differences = (
            bootstrap_samples[
                model_a
            ][metric_name]
            - bootstrap_samples[
                model_b
            ][metric_name]
        )

        pairwise_rows.append(
            {
                "ModelA": model_a,
                "ModelB": model_b,
                "K": BOOTSTRAP_K,
                "Metric": metric_name,
                "MeanDifference": float(
                    differences.mean()
                ),
                "CI95Lower": float(
                    np.quantile(
                        differences,
                        0.025,
                    )
                ),
                "CI95Upper": float(
                    np.quantile(
                        differences,
                        0.975,
                    )
                ),
                "ProbabilityDifferenceGreaterThanZero": float(
                    (
                        differences > 0
                    ).mean()
                ),
            }
        )

pairwise_results = pd.DataFrame(
    pairwise_rows
)

pairwise_results.to_csv(
    DATA_DIR
    / "bootstrap_pairwise_differences.csv",
    index=False,
)


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print(
    "\nPrimary debiased evaluation "
    f"(max inverse weight = {PRIMARY_MAX_WEIGHT:g}):"
)

print(
    debiased_results
    .round(6)
    .to_string(index=False)
)

print(
    "\nPopularity-bias evaluation:"
)

print(
    popularity_bias_results
    .round(6)
    .to_string(index=False)
)

print(
    "\nBootstrap confidence intervals at K=80:"
)

print(
    bootstrap_results
    .round(6)
    .to_string(index=False)
)

print(
    "\nPaired bootstrap differences at K=80:"
)

print(
    pairwise_results
    .round(6)
    .to_string(index=False)
)

print(
    "\nSaved:",
    DATA_DIR
    / "debiased_evaluation_comparison.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "debiased_evaluation_sensitivity.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "popularity_bias_comparison.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "recommendation_support_diagnostics.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "bootstrap_confidence_intervals.csv",
)

print(
    "Saved:",
    DATA_DIR
    / "bootstrap_pairwise_differences.csv",
)