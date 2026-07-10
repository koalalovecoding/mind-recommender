# ============================================================
# 18b - Full MIND PyTorch Learned Reranker Evaluation
#
# This process imports PyTorch but does not import FAISS.
# It loads candidates saved by 18a, applies the trained MLP,
# saves sample recommendations, evaluates dev ranking metrics,
# and builds the four-model comparison.
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"
NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MINDlarge_train"
    / "news.tsv"
)

FEATURE_NAMES = [
    "als_score",
    "log1p_popularity",
    "history_length",
    "category_affinity",
    "subcategory_affinity",
]

INPUT_DIM = len(FEATURE_NAMES)
DROPOUT = 0.10
CANDIDATE_K = 100
K_VALUES = [10, 20, 40, 80]
EVAL_BATCH_SIZE = 256

CHECKPOINT_PATH = DATA_DIR / "learned_reranker_best.pt"
FEATURE_SUMMARY_PATH = (
    DATA_DIR / "learned_reranker_feature_summary.json"
)
TRAINING_HISTORY_PATH = (
    DATA_DIR / "learned_reranker_training_history.csv"
)

EVAL_USERS_PATH = DATA_DIR / "learned_reranker_eval_users.npy"
CANDIDATES_PATH = DATA_DIR / "learned_reranker_faiss_candidates.npy"
SAMPLE_USER_PATH = DATA_DIR / "learned_reranker_sample_user.npy"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "learned_reranker_sample_candidates.npy"
SAMPLE_SCORES_PATH = DATA_DIR / "learned_reranker_sample_faiss_scores.npy"
RETRIEVAL_SUMMARY_PATH = (
    DATA_DIR / "learned_reranker_retrieval_summary.json"
)

SAMPLE_CANDIDATES_OUTPUT = (
    DATA_DIR / "learned_reranker_candidates_sample.csv"
)
SAMPLE_TOP10_OUTPUT = (
    DATA_DIR / "learned_reranker_top10.csv"
)
EVALUATION_PATH = (
    DATA_DIR / "learned_reranker_evaluation.csv"
)
COMPARISON_PATH = (
    DATA_DIR / "learned_reranker_model_comparison.csv"
)
LATENCY_PATH = (
    DATA_DIR / "learned_reranker_latency.json"
)


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def sigmoid(logits):
    """Numerically stable NumPy sigmoid for readable output."""

    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)

    positive = logits >= 0

    output[positive] = (
        1.0 / (1.0 + np.exp(-logits[positive]))
    )

    exp_values = np.exp(logits[~positive])

    output[~positive] = (
        exp_values / (1.0 + exp_values)
    )

    return output.astype(np.float32)


def load_item_metadata(item_idx_map, num_items):
    """Build item-aligned category/subcategory arrays and titles."""

    news = pd.read_csv(
        NEWS_PATH,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
        ],
        dtype="string",
    ).drop_duplicates("news_id")

    categories = sorted(
        str(value)
        for value in news["category"].dropna().unique()
    )

    subcategories = sorted(
        str(value)
        for value in news["subcategory"].dropna().unique()
    )

    category_map = {
        value: idx
        for idx, value in enumerate(categories)
    }

    subcategory_map = {
        value: idx
        for idx, value in enumerate(subcategories)
    }

    item_categories = np.full(
        num_items,
        -1,
        dtype=np.int32,
    )

    item_subcategories = np.full(
        num_items,
        -1,
        dtype=np.int32,
    )

    title_map = {}

    for row in news.itertuples(index=False):
        news_id = str(row.news_id)
        item_idx = item_idx_map.get(news_id)

        if item_idx is None:
            continue

        if pd.notna(row.category):
            item_categories[item_idx] = category_map[
                str(row.category)
            ]

        if pd.notna(row.subcategory):
            item_subcategories[item_idx] = subcategory_map[
                str(row.subcategory)
            ]

        title_map[news_id] = (
            "Title not found"
            if pd.isna(row.title)
            else str(row.title)
        )

    return (
        item_categories,
        item_subcategories,
        title_map,
        len(categories),
        len(subcategories),
    )


class MLPRanker(nn.Module):
    """Same architecture used to create the saved checkpoint."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def load_ranker(device):
    """Load the already trained best checkpoint."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "learned_reranker_best.pt was not found. "
            "Run the training script first."
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise ValueError(
            "Checkpoint feature order does not match this script."
        )

    model = MLPRanker().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    mean = np.asarray(
        checkpoint["feature_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["feature_std"],
        dtype=np.float32,
    )

    return model, mean, std, checkpoint


def make_inference_features(
    user_vector,
    candidate_items,
    seen_items,
    item_factors,
    popularity,
    item_categories,
    item_subcategories,
    num_categories,
    num_subcategories,
):
    """Construct the same five features used during training."""

    history_length = len(seen_items)

    als_scores = (
        item_factors[candidate_items]
        @ user_vector
    ).astype(np.float32)

    log_popularity = np.log1p(
        popularity[candidate_items]
    ).astype(np.float32)

    history_lengths = np.full(
        len(candidate_items),
        history_length,
        dtype=np.float32,
    )

    category_affinity = np.zeros(
        len(candidate_items),
        dtype=np.float32,
    )

    subcategory_affinity = np.zeros(
        len(candidate_items),
        dtype=np.float32,
    )

    if history_length > 0:
        history_categories = item_categories[seen_items]
        history_categories = history_categories[
            history_categories >= 0
        ]

        category_counts = np.bincount(
            history_categories,
            minlength=num_categories,
        )

        candidate_categories = item_categories[
            candidate_items
        ]

        valid_categories = candidate_categories >= 0

        category_affinity[valid_categories] = (
            category_counts[
                candidate_categories[valid_categories]
            ]
            / history_length
        )

        history_subcategories = item_subcategories[
            seen_items
        ]

        history_subcategories = history_subcategories[
            history_subcategories >= 0
        ]

        subcategory_counts = np.bincount(
            history_subcategories,
            minlength=num_subcategories,
        )

        candidate_subcategories = item_subcategories[
            candidate_items
        ]

        valid_subcategories = (
            candidate_subcategories >= 0
        )

        subcategory_affinity[valid_subcategories] = (
            subcategory_counts[
                candidate_subcategories[valid_subcategories]
            ]
            / history_length
        )

    return np.column_stack(
        [
            als_scores,
            log_popularity,
            history_lengths,
            category_affinity,
            subcategory_affinity,
        ]
    ).astype(np.float32, copy=False)


def score_features(model, raw_features, mean, std, device):
    """Standardize raw features and return MLP logits."""

    standardized = (
        raw_features - mean
    ) / std

    feature_tensor = torch.from_numpy(
        np.ascontiguousarray(
            standardized,
            dtype=np.float32,
        )
    ).to(device)

    with torch.no_grad():
        logits = model(feature_tensor)

    return (
        logits.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def evaluate_user(recommended, relevant, k):
    """Recall, NDCG, MRR, MAP, and Hit Rate for one user."""

    hits = np.isin(
        recommended[:k],
        relevant,
    ).astype(np.float64)

    recall = hits.sum() / len(relevant)

    discounts = 1.0 / np.log2(
        np.arange(2, len(hits) + 2)
    )

    dcg = np.sum(hits * discounts)
    ideal_length = min(len(relevant), k)
    idcg = np.sum(discounts[:ideal_length])
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
        [recall, ndcg, mrr, map_k, hit_rate],
        dtype=np.float64,
    )


def check_metrics():
    perfect = evaluate_user(
        np.array([1, 2]),
        np.array([1, 2]),
        2,
    )

    no_hit = evaluate_user(
        np.array([3, 4]),
        np.array([1, 2]),
        2,
    )

    assert np.allclose(perfect, 1.0)
    assert np.allclose(no_hit, 0.0)

    print("Metric sanity checks passed.")


def save_sample(
    model,
    mean,
    std,
    train_matrix,
    user_factors,
    item_factors,
    popularity,
    item_categories,
    item_subcategories,
    num_categories,
    num_subcategories,
    idx_user_map,
    idx_item_map,
    title_map,
    device,
):
    """Score and save the sample user's retrieved candidates."""

    user_idx = int(
        np.load(SAMPLE_USER_PATH)
    )

    candidates = np.load(
        SAMPLE_CANDIDATES_PATH
    ).astype(np.int64)

    faiss_scores = np.load(
        SAMPLE_SCORES_PATH
    ).astype(np.float32)

    user_vector = np.ascontiguousarray(
        user_factors[user_idx],
        dtype=np.float32,
    )

    seen_items = train_matrix[user_idx].indices

    direct_als_scores = (
        item_factors[candidates]
        @ user_vector
    )

    if not np.allclose(
        faiss_scores,
        direct_als_scores,
        atol=1e-5,
    ):
        raise ValueError(
            "Saved FAISS scores do not match ALS dot products."
        )

    raw_features = make_inference_features(
        user_vector=user_vector,
        candidate_items=candidates,
        seen_items=seen_items,
        item_factors=item_factors,
        popularity=popularity,
        item_categories=item_categories,
        item_subcategories=item_subcategories,
        num_categories=num_categories,
        num_subcategories=num_subcategories,
    )

    logits = score_features(
        model,
        raw_features,
        mean,
        std,
        device,
    )

    probabilities = sigmoid(logits)

    rows = []

    for position, item_idx in enumerate(candidates):
        item_idx = int(item_idx)
        news_id = idx_item_map[str(item_idx)]

        row = {
            "retrieval_rank": position + 1,
            "user_idx": user_idx,
            "user_id": idx_user_map[str(user_idx)],
            "item_idx": item_idx,
            "news_id": news_id,
            "title": title_map.get(
                news_id,
                "Title not found",
            ),
            "faiss_score": float(
                faiss_scores[position]
            ),
        }

        for feature_idx, feature_name in enumerate(
            FEATURE_NAMES
        ):
            row[feature_name] = float(
                raw_features[position, feature_idx]
            )

        row["learned_logit"] = float(
            logits[position]
        )

        row["learned_probability"] = float(
            probabilities[position]
        )

        rows.append(row)

    candidates_df = pd.DataFrame(rows)

    order = np.argsort(
        -logits,
        kind="stable",
    )

    top10 = candidates_df.iloc[
        order[:10]
    ].copy()

    top10.insert(
        0,
        "final_rank",
        np.arange(1, 11),
    )

    candidates_df.to_csv(
        SAMPLE_CANDIDATES_OUTPUT,
        index=False,
    )

    top10.to_csv(
        SAMPLE_TOP10_OUTPUT,
        index=False,
    )

    print("\nLearned-reranker sample top-10:")
    print(top10.to_string(index=False))


def evaluate_learned(
    model,
    mean,
    std,
    users,
    candidate_memmap,
    train_matrix,
    dev_matrix,
    user_factors,
    item_factors,
    popularity,
    item_categories,
    item_subcategories,
    num_categories,
    num_subcategories,
    device,
):
    """Score the candidates saved by 18a and evaluate dev metrics."""

    metric_sums = {
        k: np.zeros(5, dtype=np.float64)
        for k in K_VALUES
    }

    candidate_recall_sum = 0.0
    start_time = time.perf_counter()

    for start in range(
        0,
        len(users),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(users),
        )

        batch_users = users[start:end]
        batch_candidates = np.asarray(
            candidate_memmap[start:end],
            dtype=np.int64,
        )

        batch_train = train_matrix[
            batch_users
        ]

        batch_vectors = np.ascontiguousarray(
            user_factors[batch_users],
            dtype=np.float32,
        )

        feature_matrix = np.empty(
            (
                len(batch_users),
                CANDIDATE_K,
                INPUT_DIM,
            ),
            dtype=np.float32,
        )

        for row in range(len(batch_users)):
            feature_matrix[row] = make_inference_features(
                user_vector=batch_vectors[row],
                candidate_items=batch_candidates[row],
                seen_items=batch_train[row].indices,
                item_factors=item_factors,
                popularity=popularity,
                item_categories=item_categories,
                item_subcategories=item_subcategories,
                num_categories=num_categories,
                num_subcategories=num_subcategories,
            )

        flattened_logits = score_features(
            model=model,
            raw_features=feature_matrix.reshape(
                -1,
                INPUT_DIM,
            ),
            mean=mean,
            std=std,
            device=device,
        )

        batch_logits = flattened_logits.reshape(
            len(batch_users),
            CANDIDATE_K,
        )

        orders = np.argsort(
            -batch_logits,
            axis=1,
            kind="stable",
        )

        for row, user_idx in enumerate(batch_users):
            candidates = batch_candidates[row]

            recommendations = candidates[
                orders[row]
            ]

            relevant = dev_matrix[
                user_idx
            ].indices

            candidate_recall_sum += (
                np.isin(candidates, relevant).sum()
                / len(relevant)
            )

            for k in K_VALUES:
                metric_sums[k] += evaluate_user(
                    recommendations,
                    relevant,
                    k,
                )

        if end % 10_000 < EVAL_BATCH_SIZE or end == len(users):
            elapsed = time.perf_counter() - start_time

            print(
                f"Reranked {end:,}/{len(users):,} users "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    rerank_seconds = time.perf_counter() - start_time
    candidate_recall = candidate_recall_sum / len(users)

    rows = []

    for k in K_VALUES:
        (
            recall,
            ndcg,
            mrr,
            map_k,
            hit_rate,
        ) = metric_sums[k] / len(users)

        rows.append(
            {
                "Model": "TwoStageLearned",
                "K": k,
                "EvaluatedUsers": len(users),
                "Recall": recall,
                "NDCG": ndcg,
                "MRR": mrr,
                "MAP": map_k,
                "HitRate": hit_rate,
                "CandidateRecall@100": candidate_recall,
                "ALSWeight": np.nan,
                "PopularityWeight": np.nan,
            }
        )

    evaluation = pd.DataFrame(rows)

    evaluation.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    return evaluation, rerank_seconds


def build_comparison(learned_evaluation):
    """Append learned results to the Step 17 model comparison."""

    prior_path = (
        DATA_DIR / "two_stage_model_comparison.csv"
    )

    if not prior_path.exists():
        raise FileNotFoundError(
            "Run Step 17 before Step 18."
        )

    prior = pd.read_csv(prior_path)

    required_models = {
        "Popularity",
        "ALS",
        "TwoStageHeuristic",
    }

    prior = prior[
        prior["Model"].isin(required_models)
        & prior["K"].isin(K_VALUES)
    ].copy()

    expected_rows = (
        len(required_models)
        * len(K_VALUES)
    )

    if len(prior) != expected_rows:
        raise ValueError(
            "Unexpected Step 17 comparison rows."
        )

    columns = [
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

    for column in columns:
        if column not in prior:
            prior[column] = np.nan

        if column not in learned_evaluation:
            learned_evaluation[column] = np.nan

    comparison = pd.concat(
        [
            prior[columns],
            learned_evaluation[columns],
        ],
        ignore_index=True,
    )

    model_order = {
        "Popularity": 0,
        "ALS": 1,
        "TwoStageHeuristic": 2,
        "TwoStageLearned": 3,
    }

    comparison["_model_order"] = (
        comparison["Model"].map(model_order)
    )

    comparison = (
        comparison
        .sort_values(["K", "_model_order"])
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    return comparison


def build_latency(rerank_seconds, evaluated_users):
    """Combine timings produced by the separate processes."""

    retrieval = load_json(
        RETRIEVAL_SUMMARY_PATH
    )

    feature_summary = (
        load_json(FEATURE_SUMMARY_PATH)
        if FEATURE_SUMMARY_PATH.exists()
        else {}
    )

    training_seconds = None

    if TRAINING_HISTORY_PATH.exists():
        history = pd.read_csv(
            TRAINING_HISTORY_PATH
        )

        if "epoch_seconds" in history:
            training_seconds = float(
                history["epoch_seconds"].sum()
            )

    retrieval_seconds = float(
        retrieval["retrieval_seconds"]
    )

    latency = {
        "evaluated_users": int(evaluated_users),
        "feature_build_seconds": feature_summary.get(
            "feature_build_seconds"
        ),
        "ranker_training_seconds": training_seconds,
        "faiss_index_build_seconds": retrieval.get(
            "faiss_index_build_seconds"
        ),
        "retrieval_seconds": retrieval_seconds,
        "retrieval_ms_per_user": (
            1000
            * retrieval_seconds
            / evaluated_users
        ),
        "feature_and_rerank_seconds": rerank_seconds,
        "feature_and_rerank_ms_per_user": (
            1000
            * rerank_seconds
            / evaluated_users
        ),
        "end_to_end_seconds": (
            retrieval_seconds
            + rerank_seconds
        ),
        "end_to_end_ms_per_user": (
            1000
            * (
                retrieval_seconds
                + rerank_seconds
            )
            / evaluated_users
        ),
        "note": (
            "FAISS retrieval and PyTorch reranking ran in "
            "separate processes to avoid the macOS OpenMP conflict."
        ),
    }

    with open(
        LATENCY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(latency, file, indent=2)

    return latency


def main():
    torch.manual_seed(42)
    check_metrics()

    device = choose_device()

    print("PyTorch device:", device)

    required_paths = [
        EVAL_USERS_PATH,
        CANDIDATES_PATH,
        SAMPLE_USER_PATH,
        SAMPLE_CANDIDATES_PATH,
        SAMPLE_SCORES_PATH,
        RETRIEVAL_SUMMARY_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Run 18a first. Missing:\n"
            + "\n".join(missing)
        )

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
        np.load(DATA_DIR / "als_item_factors.npy"),
        dtype=np.float32,
    )

    popularity = np.load(
        DATA_DIR / "popularity_scores.npy"
    ).astype(np.float32, copy=False)

    item_idx_map = load_json(
        DATA_DIR / "item_idx_map.json"
    )

    idx_user_map = load_json(
        DATA_DIR / "idx_user_map.json"
    )

    idx_item_map = load_json(
        DATA_DIR / "idx_item_map.json"
    )

    (
        item_categories,
        item_subcategories,
        title_map,
        num_categories,
        num_subcategories,
    ) = load_item_metadata(
        item_idx_map,
        train_matrix.shape[1],
    )

    users = np.load(
        EVAL_USERS_PATH,
        mmap_mode="r",
    )

    candidate_memmap = np.load(
        CANDIDATES_PATH,
        mmap_mode="r",
    )

    if candidate_memmap.shape != (
        len(users),
        CANDIDATE_K,
    ):
        raise ValueError(
            "Saved candidate matrix has the wrong shape."
        )

    if train_matrix.shape != dev_matrix.shape:
        raise ValueError(
            "Train and dev matrix shapes do not match."
        )

    if user_factors.shape[0] != train_matrix.shape[0]:
        raise ValueError(
            "User factors do not match train users."
        )

    if item_factors.shape[0] != train_matrix.shape[1]:
        raise ValueError(
            "Item factors do not match train items."
        )

    model, mean, std, checkpoint = load_ranker(
        device
    )

    print(
        "Loaded best checkpoint: epoch",
        checkpoint["best_epoch"],
    )

    print(
        "Best validation loss:",
        checkpoint["best_validation_loss"],
    )

    save_sample(
        model=model,
        mean=mean,
        std=std,
        train_matrix=train_matrix,
        user_factors=user_factors,
        item_factors=item_factors,
        popularity=popularity,
        item_categories=item_categories,
        item_subcategories=item_subcategories,
        num_categories=num_categories,
        num_subcategories=num_subcategories,
        idx_user_map=idx_user_map,
        idx_item_map=idx_item_map,
        title_map=title_map,
        device=device,
    )

    evaluation, rerank_seconds = evaluate_learned(
        model=model,
        mean=mean,
        std=std,
        users=users,
        candidate_memmap=candidate_memmap,
        train_matrix=train_matrix,
        dev_matrix=dev_matrix,
        user_factors=user_factors,
        item_factors=item_factors,
        popularity=popularity,
        item_categories=item_categories,
        item_subcategories=item_subcategories,
        num_categories=num_categories,
        num_subcategories=num_subcategories,
        device=device,
    )

    retrieval_summary = load_json(
        RETRIEVAL_SUMMARY_PATH
    )

    saved_candidate_recall = float(
        retrieval_summary[
            "candidate_recall_at_100"
        ]
    )

    measured_candidate_recall = float(
        evaluation[
            "CandidateRecall@100"
        ].iloc[0]
    )

    if not np.isclose(
        saved_candidate_recall,
        measured_candidate_recall,
        atol=1e-12,
    ):
        raise ValueError(
            "18a and 18b candidate recall values do not match."
        )

    comparison = build_comparison(
        evaluation
    )

    latency = build_latency(
        rerank_seconds,
        len(users),
    )

    print("\nLearned reranker evaluation:")
    print(
        evaluation.round(6).to_string(
            index=False
        )
    )

    print("\nFour-model comparison:")
    print(
        comparison.round(6).to_string(
            index=False
        )
    )

    print("\nLatency:")
    print(json.dumps(latency, indent=2))


if __name__ == "__main__":
    main()
