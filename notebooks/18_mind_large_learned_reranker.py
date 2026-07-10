# ============================================================
# Step 18: Full MIND PyTorch Learned Reranker
#
# train impressions -> positive/negative samples -> five features
# -> original-order 90/10 split -> standardized PyTorch MLP
# -> FAISS top-100 -> learned reranking -> dev evaluation
# -> Popularity / ALS / Heuristic / Learned comparison
# ============================================================

import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from torch import nn
from torch.utils.data import DataLoader, IterableDataset


# -----------------------------
# Paths and configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mindlarge"
)

RAW_TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MINDlarge_train"
)

BEHAVIORS_PATH = (
    RAW_TRAIN_DIR
    / "behaviors.tsv"
)

NEWS_PATH = (
    RAW_TRAIN_DIR
    / "news.tsv"
)

FEATURE_ROOT = (
    DATA_DIR
    / "learned_reranker_features"
)

TRAIN_PART_DIR = (
    FEATURE_ROOT
    / "train"
)

VALID_PART_DIR = (
    FEATURE_ROOT
    / "validation"
)

FEATURE_SUMMARY_PATH = (
    DATA_DIR
    / "learned_reranker_feature_summary.json"
)

MEAN_PATH = (
    DATA_DIR
    / "learned_reranker_feature_mean.npy"
)

STD_PATH = (
    DATA_DIR
    / "learned_reranker_feature_std.npy"
)

CHECKPOINT_PATH = (
    DATA_DIR
    / "learned_reranker_best.pt"
)

HISTORY_PATH = (
    DATA_DIR
    / "learned_reranker_training_history.csv"
)

EVALUATION_PATH = (
    DATA_DIR
    / "learned_reranker_evaluation.csv"
)

COMPARISON_PATH = (
    DATA_DIR
    / "learned_reranker_model_comparison.csv"
)

SAMPLE_CANDIDATES_PATH = (
    DATA_DIR
    / "learned_reranker_candidates_sample.csv"
)

SAMPLE_TOP10_PATH = (
    DATA_DIR
    / "learned_reranker_top10.csv"
)

LATENCY_PATH = (
    DATA_DIR
    / "learned_reranker_latency.json"
)


# -----------------------------
# Sampling and split settings
# -----------------------------

SEED = 42

# Keep every clicked candidate.
# For each positive, sample at most four exposed non-clicked items
# from the same impression.
NEGATIVE_RATIO = 4

# First 90% of official train behavior rows:
#     ranker training
#
# Last 10%:
#     internal ranker validation
#
# This preserves original row order and avoids random splitting.
TRAIN_FRACTION = 0.90

# Number of candidate-level samples in each NumPy part file.
PART_ROWS = 500_000

# False:
#     reuse existing feature parts when they exist.
#
# True:
#     delete and rebuild all ranker samples.
#
# Set this to True after changing feature definitions,
# NEGATIVE_RATIO, or TRAIN_FRACTION.
REBUILD_FEATURES = False


# -----------------------------
# Ranking features
# -----------------------------

FEATURE_NAMES = [
    "als_score",
    "log1p_popularity",
    "history_length",
    "category_affinity",
    "subcategory_affinity",
]

INPUT_DIM = len(FEATURE_NAMES)


# -----------------------------
# PyTorch settings
# -----------------------------

TRAIN_BATCH_SIZE = 8192

MAX_EPOCHS = 8

PATIENCE = 2

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

DROPOUT = 0.10


# -----------------------------
# Retrieval and evaluation
# -----------------------------

CANDIDATE_K = 100

K_VALUES = [
    10,
    20,
    40,
    80,
]

EVAL_BATCH_SIZE = 256

SAMPLE_USER_IDX = 0


# -----------------------------
# Optional debugging limits
# -----------------------------
#
# Keep both as None for the real Full MIND experiment.

DEBUG_MAX_BEHAVIOR_ROWS = None

DEBUG_MAX_EVAL_USERS = None


# ============================================================
# General helpers
# ============================================================

def set_seed(seed):
    """Set deterministic Python, NumPy, and PyTorch seeds."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    """
    Prefer CUDA, then Apple Silicon MPS, then CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def load_json(path):
    """Load one UTF-8 JSON file."""

    with open(
        path,
        encoding="utf-8",
    ) as file:
        return json.load(file)


def count_lines(path):
    """
    Count behavior rows without loading the file into memory.
    """

    with open(
        path,
        "rb",
    ) as file:
        return sum(
            1
            for _ in file
        )


def sigmoid(logits):
    """
    Numerically stable NumPy sigmoid.

    The model is ranked by logits. Probabilities are saved only
    to make the sample recommendation output easier to inspect.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    output = np.empty_like(
        logits
    )

    positive_mask = (
        logits >= 0
    )

    output[positive_mask] = (
        1.0
        / (
            1.0
            + np.exp(
                -logits[positive_mask]
            )
        )
    )

    negative_logits = logits[
        ~positive_mask
    ]

    exp_values = np.exp(
        negative_logits
    )

    output[~positive_mask] = (
        exp_values
        / (
            1.0
            + exp_values
        )
    )

    return output.astype(
        np.float32
    )


# ============================================================
# News metadata aligned with item_idx
# ============================================================

def load_item_metadata(
    item_idx_map,
    num_items,
):
    """
    Build integer category/subcategory arrays aligned with item_idx.

    Returns
    -------
    item_categories:
        Array where item_categories[item_idx] gives the item's
        integer category code.

    item_subcategories:
        Array where item_subcategories[item_idx] gives the item's
        integer subcategory code.

    title_map:
        raw news_id -> title.

    num_categories / num_subcategories:
        Used for fast user-profile construction during evaluation.
    """

    news = pd.read_csv(
        NEWS_PATH,
        sep="\t",
        header=None,
        usecols=[
            0,
            1,
            2,
            3,
        ],
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
        ],
        dtype="string",
    ).drop_duplicates(
        "news_id"
    )

    categories = sorted(
        str(value)
        for value in news[
            "category"
        ].dropna().unique()
    )

    subcategories = sorted(
        str(value)
        for value in news[
            "subcategory"
        ].dropna().unique()
    )

    category_map = {
        value: idx
        for idx, value in enumerate(
            categories
        )
    }

    subcategory_map = {
        value: idx
        for idx, value in enumerate(
            subcategories
        )
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

    for row in news.itertuples(
        index=False
    ):
        news_id = str(
            row.news_id
        )

        item_idx = item_idx_map.get(
            news_id
        )

        if item_idx is None:
            continue

        if pd.notna(
            row.category
        ):
            item_categories[
                item_idx
            ] = category_map[
                str(row.category)
            ]

        if pd.notna(
            row.subcategory
        ):
            item_subcategories[
                item_idx
            ] = subcategory_map[
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


# ============================================================
# Memory-safe feature-part writer
# ============================================================

class PartWriter:
    """
    Save fixed-size feature and label arrays as .npy parts.

    Preallocated NumPy buffers avoid storing millions of Python
    feature lists in memory.
    """

    def __init__(
        self,
        output_dir,
        part_rows,
    ):
        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.part_rows = (
            part_rows
        )

        self.features = np.empty(
            (
                part_rows,
                INPUT_DIM,
            ),
            dtype=np.float32,
        )

        self.labels = np.empty(
            part_rows,
            dtype=np.uint8,
        )

        self.position = 0

        self.part_number = 0

        self.total_rows = 0

    def add(
        self,
        features,
        labels,
    ):
        """Append a vectorized candidate block."""

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        labels = np.asarray(
            labels,
            dtype=np.uint8,
        )

        if (
            features.ndim != 2
            or features.shape[1] != INPUT_DIM
        ):
            raise ValueError(
                "Unexpected feature-array shape."
            )

        if len(features) != len(labels):
            raise ValueError(
                "Feature and label counts do not match."
            )

        source_position = 0

        while source_position < len(
            labels
        ):
            available = (
                self.part_rows
                - self.position
            )

            take = min(
                available,
                len(labels)
                - source_position,
            )

            target_slice = slice(
                self.position,
                self.position + take,
            )

            source_slice = slice(
                source_position,
                source_position + take,
            )

            self.features[
                target_slice
            ] = features[
                source_slice
            ]

            self.labels[
                target_slice
            ] = labels[
                source_slice
            ]

            self.position += take

            self.total_rows += take

            source_position += take

            if (
                self.position
                == self.part_rows
            ):
                self.flush()

    def flush(self):
        """Save the currently filled part."""

        if self.position == 0:
            return

        feature_path = (
            self.output_dir
            / (
                f"features-"
                f"{self.part_number:05d}.npy"
            )
        )

        label_path = (
            self.output_dir
            / (
                f"labels-"
                f"{self.part_number:05d}.npy"
            )
        )

        np.save(
            feature_path,
            self.features[
                :self.position
            ],
            allow_pickle=False,
        )

        np.save(
            label_path,
            self.labels[
                :self.position
            ],
            allow_pickle=False,
        )

        print(
            f"Saved {self.output_dir.name} "
            f"part {self.part_number:05d}: "
            f"{self.position:,} samples",
            flush=True,
        )

        self.part_number += 1

        self.position = 0

    def close(self):
        """Save the final partially filled part."""

        self.flush()


# ============================================================
# Training-sample parsing
# ============================================================

def parse_candidates(
    impression_string,
    item_idx_map,
):
    """
    Parse one MIND impression.

    Nxxxxx-1:
        exposed and clicked positive.

    Nxxxxx-0:
        exposed but not clicked negative candidate.
    """

    positives = []

    negatives = []

    skipped = 0

    for token in impression_string.split():
        try:
            news_id, label = token.rsplit(
                "-",
                1,
            )
        except ValueError:
            skipped += 1
            continue

        item_idx = item_idx_map.get(
            news_id
        )

        if item_idx is None:
            skipped += 1

        elif label == "1":
            positives.append(
                item_idx
            )

        elif label == "0":
            negatives.append(
                item_idx
            )

        else:
            skipped += 1

    return (
        np.asarray(
            positives,
            dtype=np.int64,
        ),
        np.asarray(
            negatives,
            dtype=np.int64,
        ),
        skipped,
    )


def build_history_profile(
    history_string,
    item_idx_map,
    item_categories,
    item_subcategories,
):
    """
    Construct the profile available before the current impression.

    Category affinity:

        historical clicks in candidate category
        ----------------------------------------
                    history length

    Subcategory affinity is defined analogously.

    Using the current impression's history avoids inserting the
    candidate's current label or later user behavior into affinity
    features.
    """

    category_counts = defaultdict(
        int
    )

    subcategory_counts = defaultdict(
        int
    )

    history_length = 0

    skipped = 0

    history_news_ids = (
        history_string.split()
        if history_string
        else []
    )

    for news_id in history_news_ids:
        item_idx = item_idx_map.get(
            news_id
        )

        if item_idx is None:
            skipped += 1
            continue

        history_length += 1

        category = int(
            item_categories[
                item_idx
            ]
        )

        subcategory = int(
            item_subcategories[
                item_idx
            ]
        )

        if category >= 0:
            category_counts[
                category
            ] += 1

        if subcategory >= 0:
            subcategory_counts[
                subcategory
            ] += 1

    return (
        history_length,
        category_counts,
        subcategory_counts,
        skipped,
    )


def make_training_features(
    user_idx,
    candidate_items,
    history_length,
    category_counts,
    subcategory_counts,
    user_factors,
    item_factors,
    popularity,
    item_categories,
    item_subcategories,
):
    """
    Construct the five raw candidate-level ranker features.
    """

    user_vector = np.asarray(
        user_factors[
            user_idx
        ],
        dtype=np.float32,
    )

    als_scores = (
        item_factors[
            candidate_items
        ]
        @ user_vector
    ).astype(
        np.float32
    )

    log_popularity = np.log1p(
        popularity[
            candidate_items
        ]
    ).astype(
        np.float32
    )

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
        inverse_length = (
            1.0
            / history_length
        )

        for (
            position,
            item_idx,
        ) in enumerate(
            candidate_items
        ):
            category = int(
                item_categories[
                    item_idx
                ]
            )

            subcategory = int(
                item_subcategories[
                    item_idx
                ]
            )

            if category >= 0:
                category_affinity[
                    position
                ] = (
                    category_counts.get(
                        category,
                        0,
                    )
                    * inverse_length
                )

            if subcategory >= 0:
                subcategory_affinity[
                    position
                ] = (
                    subcategory_counts.get(
                        subcategory,
                        0,
                    )
                    * inverse_length
                )

    return np.column_stack(
        [
            als_scores,
            log_popularity,
            history_lengths,
            category_affinity,
            subcategory_affinity,
        ]
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Build ranker train/validation feature files
# ============================================================

def can_reuse_features():
    """
    Check whether generated feature files already exist.
    """

    return (
        not REBUILD_FEATURES
        and FEATURE_SUMMARY_PATH.exists()
        and MEAN_PATH.exists()
        and STD_PATH.exists()
        and any(
            TRAIN_PART_DIR.glob(
                "features-*.npy"
            )
        )
        and any(
            VALID_PART_DIR.glob(
                "features-*.npy"
            )
        )
    )


def build_feature_parts(
    user_idx_map,
    item_idx_map,
    user_factors,
    item_factors,
    popularity,
    item_categories,
    item_subcategories,
):
    """
    Stream Full MIND official train impressions.

    The function:

    1. keeps every mapped positive candidate;
    2. samples negatives from the same impression;
    3. splits by original behavior-row order;
    4. saves memory-safe feature parts;
    5. calculates feature mean/std from ranker train only.
    """

    if can_reuse_features():
        print(
            "Reusing existing learned-reranker "
            "feature parts."
        )

        return (
            load_json(
                FEATURE_SUMMARY_PATH
            ),
            np.load(
                MEAN_PATH
            ),
            np.load(
                STD_PATH
            ),
        )

    if FEATURE_ROOT.exists():
        shutil.rmtree(
            FEATURE_ROOT
        )

    total_rows = count_lines(
        BEHAVIORS_PATH
    )

    if (
        DEBUG_MAX_BEHAVIOR_ROWS
        is not None
    ):
        total_rows = min(
            total_rows,
            DEBUG_MAX_BEHAVIOR_ROWS,
        )

    split_row = int(
        total_rows
        * TRAIN_FRACTION
    )

    if (
        split_row <= 0
        or split_row >= total_rows
    ):
        raise ValueError(
            "The ranker train/validation "
            "split is empty."
        )

    print(
        f"Behavior rows used: "
        f"{total_rows:,}"
    )

    print(
        f"Ranker train rows: "
        f"{split_row:,}"
    )

    print(
        f"Ranker validation rows: "
        f"{total_rows - split_row:,}"
    )

    train_writer = PartWriter(
        TRAIN_PART_DIR,
        PART_ROWS,
    )

    valid_writer = PartWriter(
        VALID_PART_DIR,
        PART_ROWS,
    )

    rng = np.random.default_rng(
        SEED
    )

    feature_sum = np.zeros(
        INPUT_DIM,
        dtype=np.float64,
    )

    feature_square_sum = np.zeros(
        INPUT_DIM,
        dtype=np.float64,
    )

    train_sample_count = 0

    counts = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }

    start_time = time.perf_counter()

    with open(
        BEHAVIORS_PATH,
        encoding="utf-8",
    ) as file:
        for (
            row_number,
            line,
        ) in enumerate(file):

            if row_number >= total_rows:
                break

            split = (
                "train"
                if row_number < split_row
                else "validation"
            )

            writer = (
                train_writer
                if split == "train"
                else valid_writer
            )

            counts[
                split
            ][
                "behavior_rows"
            ] += 1

            fields = (
                line.rstrip("\n")
                .split("\t")
            )

            if len(fields) != 5:
                counts[
                    split
                ][
                    "malformed_rows"
                ] += 1

                continue

            (
                _impression_id,
                user_id,
                _time_string,
                history_string,
                impression_string,
            ) = fields

            user_idx = user_idx_map.get(
                user_id
            )

            if user_idx is None:
                counts[
                    split
                ][
                    "unmapped_users"
                ] += 1

                continue

            (
                positives,
                negatives,
                skipped_candidates,
            ) = parse_candidates(
                impression_string,
                item_idx_map,
            )

            counts[
                split
            ][
                "skipped_candidate_tokens"
            ] += skipped_candidates

            if len(positives) == 0:
                counts[
                    split
                ][
                    "rows_without_mapped_positive"
                ] += 1

                continue

            negative_count = min(
                len(negatives),
                NEGATIVE_RATIO
                * len(positives),
            )

            if negative_count > 0:
                sampled_negatives = rng.choice(
                    negatives,
                    size=negative_count,
                    replace=False,
                )

            else:
                sampled_negatives = np.empty(
                    0,
                    dtype=np.int64,
                )

            candidate_items = np.concatenate(
                [
                    positives,
                    sampled_negatives,
                ]
            )

            labels = np.concatenate(
                [
                    np.ones(
                        len(positives),
                        dtype=np.uint8,
                    ),
                    np.zeros(
                        len(sampled_negatives),
                        dtype=np.uint8,
                    ),
                ]
            )

            (
                history_length,
                category_counts,
                subcategory_counts,
                skipped_history,
            ) = build_history_profile(
                history_string,
                item_idx_map,
                item_categories,
                item_subcategories,
            )

            counts[
                split
            ][
                "skipped_history_items"
            ] += skipped_history

            features = make_training_features(
                user_idx=user_idx,
                candidate_items=candidate_items,
                history_length=history_length,
                category_counts=category_counts,
                subcategory_counts=subcategory_counts,
                user_factors=user_factors,
                item_factors=item_factors,
                popularity=popularity,
                item_categories=item_categories,
                item_subcategories=(
                    item_subcategories
                ),
            )

            writer.add(
                features,
                labels,
            )

            counts[
                split
            ][
                "positive_samples"
            ] += len(positives)

            counts[
                split
            ][
                "negative_samples"
            ] += len(
                sampled_negatives
            )

            counts[
                split
            ][
                "total_samples"
            ] += len(labels)

            # Standardization statistics are computed only
            # from ranker-training samples.
            if split == "train":
                feature_sum += features.sum(
                    axis=0,
                    dtype=np.float64,
                )

                feature_square_sum += np.square(
                    features,
                    dtype=np.float64,
                ).sum(
                    axis=0
                )

                train_sample_count += len(
                    features
                )

            completed = (
                row_number + 1
            )

            if completed % 100_000 == 0:
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Processed "
                    f"{completed:,}/"
                    f"{total_rows:,} rows "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )

    train_writer.close()

    valid_writer.close()

    if (
        train_sample_count == 0
        or valid_writer.total_rows == 0
    ):
        raise ValueError(
            "No ranker train or validation "
            "samples were generated."
        )

    mean = (
        feature_sum
        / train_sample_count
    )

    variance = np.maximum(
        feature_square_sum
        / train_sample_count
        - mean ** 2,
        0.0,
    )

    std = np.sqrt(
        variance
    )

    # Prevent division by zero for constant features.
    std[
        std < 1e-6
    ] = 1.0

    mean = mean.astype(
        np.float32
    )

    std = std.astype(
        np.float32
    )

    np.save(
        MEAN_PATH,
        mean,
        allow_pickle=False,
    )

    np.save(
        STD_PATH,
        std,
        allow_pickle=False,
    )

    summary = {
        "negative_ratio": (
            NEGATIVE_RATIO
        ),
        "train_fraction": (
            TRAIN_FRACTION
        ),
        "feature_names": (
            FEATURE_NAMES
        ),
        "behavior_rows_used": (
            total_rows
        ),
        "train_behavior_rows": (
            split_row
        ),
        "validation_behavior_rows": (
            total_rows
            - split_row
        ),
        "train": dict(
            counts["train"]
        ),
        "validation": dict(
            counts["validation"]
        ),
        "train_parts": (
            train_writer.part_number
        ),
        "validation_parts": (
            valid_writer.part_number
        ),
        "feature_mean": dict(
            zip(
                FEATURE_NAMES,
                map(float, mean),
            )
        ),
        "feature_std": dict(
            zip(
                FEATURE_NAMES,
                map(float, std),
            )
        ),
        "feature_build_seconds": (
            time.perf_counter()
            - start_time
        ),
        "note": (
            "Feature standardization uses "
            "ranker-train samples only. "
            "MIND dev labels are not used."
        ),
    }

    with open(
        FEATURE_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return (
        summary,
        mean,
        std,
    )


# ============================================================
# Mini-batch IterableDataset
# ============================================================

class PartBatchDataset(
    IterableDataset
):
    """
    Load one NumPy part at a time and yield mini-batches.

    DataLoader uses batch_size=None because this dataset already
    returns vectorized mini-batches. This avoids millions of
    per-sample Python __getitem__ calls.
    """

    def __init__(
        self,
        part_dir,
        mean,
        std,
        batch_size,
        shuffle,
        seed,
    ):
        super().__init__()

        self.paths = sorted(
            part_dir.glob(
                "features-*.npy"
            )
        )

        if not self.paths:
            raise FileNotFoundError(
                f"No feature parts found "
                f"in {part_dir}"
            )

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        )

        self.std = np.asarray(
            std,
            dtype=np.float32,
        )

        self.batch_size = (
            batch_size
        )

        self.shuffle = shuffle

        self.seed = seed

        self.epoch = 0

    def set_epoch(
        self,
        epoch,
    ):
        """Change the deterministic shuffle between epochs."""

        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed
            + self.epoch
        )

        paths = list(
            self.paths
        )

        if self.shuffle:
            rng.shuffle(
                paths
            )

        for feature_path in paths:
            label_path = feature_path.with_name(
                feature_path.name.replace(
                    "features-",
                    "labels-",
                )
            )

            features = np.load(
                feature_path,
                mmap_mode="r",
            )

            labels = np.load(
                label_path,
                mmap_mode="r",
            )

            if len(features) != len(labels):
                raise ValueError(
                    "Feature and label part "
                    "sizes do not match."
                )

            if self.shuffle:
                order = rng.permutation(
                    len(labels)
                )

            else:
                order = None

            for start in range(
                0,
                len(labels),
                self.batch_size,
            ):
                end = min(
                    start
                    + self.batch_size,
                    len(labels),
                )

                if order is None:
                    batch_features = np.asarray(
                        features[
                            start:end
                        ],
                        dtype=np.float32,
                    )

                    batch_labels = np.asarray(
                        labels[
                            start:end
                        ],
                        dtype=np.float32,
                    )

                else:
                    indices = order[
                        start:end
                    ]

                    batch_features = np.asarray(
                        features[
                            indices
                        ],
                        dtype=np.float32,
                    )

                    batch_labels = np.asarray(
                        labels[
                            indices
                        ],
                        dtype=np.float32,
                    )

                batch_features = (
                    batch_features
                    - self.mean
                ) / self.std

                yield (
                    torch.from_numpy(
                        np.ascontiguousarray(
                            batch_features
                        )
                    ),
                    torch.from_numpy(
                        np.ascontiguousarray(
                            batch_labels
                        )
                    ),
                )


# ============================================================
# PyTorch MLP ranker
# ============================================================

class MLPRanker(
    nn.Module
):
    """
    Small pointwise binary ranker.

    BCEWithLogitsLoss operates directly on the final logits.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                32,
                16,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        features,
    ):
        return self.network(
            features
        ).squeeze(
            -1
        )


def run_epoch(
    model,
    loader,
    loss_function,
    device,
    optimizer=None,
):
    """
    Run one training or validation epoch.
    """

    training = (
        optimizer
        is not None
    )

    model.train(
        training
    )

    total_loss = 0.0

    total_samples = 0

    for (
        features,
        labels,
    ) in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            logits = model(
                features
            )

            loss = loss_function(
                logits,
                labels,
            )

            if training:
                loss.backward()

                optimizer.step()

        batch_size = len(
            labels
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    if total_samples == 0:
        raise ValueError(
            "An epoch contained zero samples."
        )

    return (
        total_loss
        / total_samples
    )


def train_ranker(
    mean,
    std,
    device,
):
    """
    Train the MLP and save the checkpoint with the lowest
    validation BCE loss.
    """

    train_dataset = PartBatchDataset(
        part_dir=TRAIN_PART_DIR,
        mean=mean,
        std=std,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    valid_dataset = PartBatchDataset(
        part_dir=VALID_PART_DIR,
        mean=mean,
        std=std,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    model = MLPRanker().to(
        device
    )

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=(
            WEIGHT_DECAY
        ),
    )

    best_loss = math.inf

    stale_epochs = 0

    history_rows = []

    start_time = (
        time.perf_counter()
    )

    print(
        "Training learned reranker "
        f"on device: {device}"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_start = (
            time.perf_counter()
        )

        train_dataset.set_epoch(
            epoch
        )

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            loss_function=(
                loss_function
            ),
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            valid_loss = run_epoch(
                model=model,
                loader=valid_loader,
                loss_function=(
                    loss_function
                ),
                device=device,
                optimizer=None,
            )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_loss
                ),
                "validation_loss": (
                    valid_loss
                ),
                "epoch_seconds": (
                    epoch_seconds
                ),
            }
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train={train_loss:.6f}, "
            f"validation={valid_loss:.6f}, "
            f"seconds={epoch_seconds:.2f}"
        )

        if (
            valid_loss
            < best_loss - 1e-7
        ):
            best_loss = valid_loss

            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "feature_names": (
                        FEATURE_NAMES
                    ),
                    "feature_mean": (
                        np.asarray(
                            mean,
                            dtype=np.float32,
                        )
                    ),
                    "feature_std": (
                        np.asarray(
                            std,
                            dtype=np.float32,
                        )
                    ),
                    "best_epoch": (
                        epoch
                    ),
                    "best_validation_loss": (
                        best_loss
                    ),
                    "negative_ratio": (
                        NEGATIVE_RATIO
                    ),
                    "train_fraction": (
                        TRAIN_FRACTION
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  Saved best checkpoint."
            )

        else:
            stale_epochs += 1

            if (
                stale_epochs
                >= PATIENCE
            ):
                print(
                    "Early stopping."
                )

                break

    pd.DataFrame(
        history_rows
    ).to_csv(
        HISTORY_PATH,
        index=False,
    )

    return (
        time.perf_counter()
        - start_time
    )


def load_ranker(
    device,
):
    """
    Load the best model checkpoint.

    weights_only=False is explicitly used for modern PyTorch because
    the checkpoint also stores NumPy feature statistics.
    """

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    if (
        checkpoint[
            "feature_names"
        ]
        != FEATURE_NAMES
    ):
        raise ValueError(
            "Checkpoint feature order "
            "does not match the code."
        )

    model = MLPRanker().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    mean = np.asarray(
        checkpoint[
            "feature_mean"
        ],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint[
            "feature_std"
        ],
        dtype=np.float32,
    )

    return (
        model,
        mean,
        std,
        checkpoint,
    )


# ============================================================
# FAISS candidate retrieval
# ============================================================

def get_unseen_candidates(
    items,
    scores,
    seen_items,
):
    """
    Filter train-seen items and retain exactly 100 unique items.
    """

    valid_mask = (
        items >= 0
    )

    items = items[
        valid_mask
    ]

    scores = scores[
        valid_mask
    ]

    unseen_mask = ~np.isin(
        items,
        seen_items,
    )

    items = items[
        unseen_mask
    ][
        :CANDIDATE_K
    ]

    scores = scores[
        unseen_mask
    ][
        :CANDIDATE_K
    ]

    if len(items) != CANDIDATE_K:
        raise ValueError(
            "Could not retrieve 100 "
            "unseen candidates."
        )

    if (
        len(np.unique(items))
        != CANDIDATE_K
    ):
        raise ValueError(
            "Candidate items contain duplicates."
        )

    if np.intersect1d(
        items,
        seen_items,
    ).size:
        raise ValueError(
            "A train-seen item remains "
            "in the candidate set."
        )

    return (
        items.astype(
            np.int64,
            copy=False,
        ),
        scores.astype(
            np.float32,
            copy=False,
        ),
    )


# ============================================================
# Dev-time inference features
# ============================================================

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
    """
    Build features for one user's FAISS candidates.

    At official dev time, all official train interactions are
    historical information, so train_matrix[user] is used as the
    user's available history.

    No dev click labels are used here.
    """

    history_length = len(
        seen_items
    )

    als_scores = (
        item_factors[
            candidate_items
        ]
        @ user_vector
    ).astype(
        np.float32
    )

    log_popularity = np.log1p(
        popularity[
            candidate_items
        ]
    ).astype(
        np.float32
    )

    history_lengths = np.full(
        CANDIDATE_K,
        history_length,
        dtype=np.float32,
    )

    category_affinity = np.zeros(
        CANDIDATE_K,
        dtype=np.float32,
    )

    subcategory_affinity = np.zeros(
        CANDIDATE_K,
        dtype=np.float32,
    )

    if history_length > 0:
        history_categories = (
            item_categories[
                seen_items
            ]
        )

        history_categories = (
            history_categories[
                history_categories >= 0
            ]
        )

        category_counts = np.bincount(
            history_categories,
            minlength=(
                num_categories
            ),
        )

        candidate_categories = (
            item_categories[
                candidate_items
            ]
        )

        valid_category_mask = (
            candidate_categories
            >= 0
        )

        category_affinity[
            valid_category_mask
        ] = (
            category_counts[
                candidate_categories[
                    valid_category_mask
                ]
            ]
            / history_length
        )

        history_subcategories = (
            item_subcategories[
                seen_items
            ]
        )

        history_subcategories = (
            history_subcategories[
                history_subcategories
                >= 0
            ]
        )

        subcategory_counts = np.bincount(
            history_subcategories,
            minlength=(
                num_subcategories
            ),
        )

        candidate_subcategories = (
            item_subcategories[
                candidate_items
            ]
        )

        valid_subcategory_mask = (
            candidate_subcategories
            >= 0
        )

        subcategory_affinity[
            valid_subcategory_mask
        ] = (
            subcategory_counts[
                candidate_subcategories[
                    valid_subcategory_mask
                ]
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
    ).astype(
        np.float32,
        copy=False,
    )


def score_features(
    model,
    raw_features,
    mean,
    std,
    device,
):
    """
    Standardize raw features and calculate MLP logits.
    """

    standardized = (
        raw_features
        - mean
    ) / std

    feature_tensor = torch.from_numpy(
        np.ascontiguousarray(
            standardized,
            dtype=np.float32,
        )
    ).to(
        device
    )

    with torch.no_grad():
        logits = model(
            feature_tensor
        )

    return (
        logits
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


# ============================================================
# Ranking metrics
# ============================================================

def evaluate_user(
    recommended,
    relevant,
    k,
):
    """
    Calculate Recall@K, NDCG@K, MRR@K, MAP@K, and Hit Rate@K.
    """

    hits = np.isin(
        recommended[
            :k
        ],
        relevant,
    ).astype(
        np.float64
    )

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
        hits
        * discounts
    )

    ideal_length = min(
        len(relevant),
        k,
    )

    idcg = np.sum(
        discounts[
            :ideal_length
        ]
    )

    ndcg = (
        dcg
        / idcg
    )

    # MRR@K
    hit_positions = np.flatnonzero(
        hits
    )

    mrr = (
        0.0
        if len(hit_positions) == 0
        else 1.0
        / (
            hit_positions[0]
            + 1
        )
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
        ],
        dtype=np.float64,
    )


def check_metrics():
    """Verify perfect and no-hit examples."""

    perfect = evaluate_user(
        recommended=np.array(
            [1, 2]
        ),
        relevant=np.array(
            [1, 2]
        ),
        k=2,
    )

    no_hit = evaluate_user(
        recommended=np.array(
            [3, 4]
        ),
        relevant=np.array(
            [1, 2]
        ),
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


# ============================================================
# Readable sample recommendations
# ============================================================

def save_sample(
    index,
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
    """
    Save one user's complete top-100 candidate table and final
    learned-reranked top-10 table.
    """

    user_idx = (
        SAMPLE_USER_IDX
    )

    if (
        train_matrix[
            user_idx
        ].nnz == 0
    ):
        user_idx = int(
            np.flatnonzero(
                np.diff(
                    train_matrix.indptr
                ) > 0
            )[0]
        )

    user_vector = np.ascontiguousarray(
        user_factors[
            user_idx
        ],
        dtype=np.float32,
    )

    seen_items = train_matrix[
        user_idx
    ].indices

    search_k = min(
        train_matrix.shape[1],
        CANDIDATE_K
        + len(seen_items),
    )

    (
        raw_scores,
        raw_items,
    ) = index.search(
        user_vector[
            None,
            :
        ],
        search_k,
    )

    (
        candidates,
        faiss_scores,
    ) = get_unseen_candidates(
        raw_items[0],
        raw_scores[0],
        seen_items,
    )

    direct_als_scores = (
        item_factors[
            candidates
        ]
        @ user_vector
    )

    if not np.allclose(
        faiss_scores,
        direct_als_scores,
        atol=1e-5,
    ):
        raise ValueError(
            "FAISS scores do not match "
            "ALS factor dot products."
        )

    raw_features = (
        make_inference_features(
            user_vector=user_vector,
            candidate_items=candidates,
            seen_items=seen_items,
            item_factors=item_factors,
            popularity=popularity,
            item_categories=(
                item_categories
            ),
            item_subcategories=(
                item_subcategories
            ),
            num_categories=(
                num_categories
            ),
            num_subcategories=(
                num_subcategories
            ),
        )
    )

    logits = score_features(
        model=model,
        raw_features=raw_features,
        mean=mean,
        std=std,
        device=device,
    )

    probabilities = sigmoid(
        logits
    )

    order = np.argsort(
        -logits,
        kind="stable",
    )

    rows = []

    for (
        position,
        item_idx,
    ) in enumerate(
        candidates
    ):
        item_idx = int(
            item_idx
        )

        news_id = idx_item_map[
            str(item_idx)
        ]

        row = {
            "retrieval_rank": (
                position + 1
            ),
            "user_idx": (
                user_idx
            ),
            "user_id": (
                idx_user_map[
                    str(user_idx)
                ]
            ),
            "item_idx": (
                item_idx
            ),
            "news_id": (
                news_id
            ),
            "title": (
                title_map.get(
                    news_id,
                    "Title not found",
                )
            ),
            "faiss_score": float(
                faiss_scores[
                    position
                ]
            ),
        }

        for (
            feature_idx,
            feature_name,
        ) in enumerate(
            FEATURE_NAMES
        ):
            row[
                feature_name
            ] = float(
                raw_features[
                    position,
                    feature_idx,
                ]
            )

        row[
            "learned_logit"
        ] = float(
            logits[
                position
            ]
        )

        row[
            "learned_probability"
        ] = float(
            probabilities[
                position
            ]
        )

        rows.append(
            row
        )

    candidates_df = pd.DataFrame(
        rows
    )

    top10 = candidates_df.iloc[
        order[
            :10
        ]
    ].copy()

    top10.insert(
        0,
        "final_rank",
        np.arange(
            1,
            11,
        ),
    )

    candidates_df.to_csv(
        SAMPLE_CANDIDATES_PATH,
        index=False,
    )

    top10.to_csv(
        SAMPLE_TOP10_PATH,
        index=False,
    )

    print(
        "\nLearned-reranker sample top-10:"
    )

    print(
        top10.to_string(
            index=False
        )
    )


# ============================================================
# Untouched dev evaluation
# ============================================================

def evaluate_learned(
    index,
    model,
    mean,
    std,
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
    """
    Evaluate on users with at least one valid official dev positive.

    Dev labels are used only after:
        1. candidate retrieval;
        2. feature construction;
        3. MLP scoring;
        4. final candidate sorting.
    """

    users = np.flatnonzero(
        np.diff(
            dev_matrix.indptr
        ) > 0
    )

    if (
        DEBUG_MAX_EVAL_USERS
        is not None
    ):
        users = users[
            :DEBUG_MAX_EVAL_USERS
        ]

    if len(users) == 0:
        raise ValueError(
            "No evaluable dev users were found."
        )

    metric_sums = {
        k: np.zeros(
            5,
            dtype=np.float64,
        )
        for k in K_VALUES
    }

    candidate_recall_sum = 0.0

    retrieval_seconds = 0.0

    rerank_seconds = 0.0

    total_start = (
        time.perf_counter()
    )

    for start in range(
        0,
        len(users),
        EVAL_BATCH_SIZE,
    ):
        batch_users = users[
            start:
            start
            + EVAL_BATCH_SIZE
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
            CANDIDATE_K
            + max_seen,
        )

        retrieval_start = (
            time.perf_counter()
        )

        (
            batch_scores,
            batch_items,
        ) = index.search(
            batch_vectors,
            search_k,
        )

        retrieval_seconds += (
            time.perf_counter()
            - retrieval_start
        )

        rerank_start = (
            time.perf_counter()
        )

        candidate_matrix = np.empty(
            (
                len(batch_users),
                CANDIDATE_K,
            ),
            dtype=np.int64,
        )

        feature_matrix = np.empty(
            (
                len(batch_users),
                CANDIDATE_K,
                INPUT_DIM,
            ),
            dtype=np.float32,
        )

        for row in range(
            len(batch_users)
        ):
            seen_items = (
                batch_train[
                    row
                ].indices
            )

            (
                candidates,
                _candidate_scores,
            ) = get_unseen_candidates(
                batch_items[
                    row
                ],
                batch_scores[
                    row
                ],
                seen_items,
            )

            candidate_matrix[
                row
            ] = candidates

            feature_matrix[
                row
            ] = make_inference_features(
                user_vector=(
                    batch_vectors[
                        row
                    ]
                ),
                candidate_items=(
                    candidates
                ),
                seen_items=(
                    seen_items
                ),
                item_factors=(
                    item_factors
                ),
                popularity=(
                    popularity
                ),
                item_categories=(
                    item_categories
                ),
                item_subcategories=(
                    item_subcategories
                ),
                num_categories=(
                    num_categories
                ),
                num_subcategories=(
                    num_subcategories
                ),
            )

        flattened_features = (
            feature_matrix.reshape(
                -1,
                INPUT_DIM,
            )
        )

        flattened_logits = score_features(
            model=model,
            raw_features=(
                flattened_features
            ),
            mean=mean,
            std=std,
            device=device,
        )

        batch_logits = (
            flattened_logits.reshape(
                len(batch_users),
                CANDIDATE_K,
            )
        )

        orders = np.argsort(
            -batch_logits,
            axis=1,
            kind="stable",
        )

        rerank_seconds += (
            time.perf_counter()
            - rerank_start
        )

        for (
            row,
            user_idx,
        ) in enumerate(
            batch_users
        ):
            candidates = (
                candidate_matrix[
                    row
                ]
            )

            recommendations = candidates[
                orders[
                    row
                ]
            ]

            relevant = dev_matrix[
                user_idx
            ].indices

            candidate_recall_sum += (
                np.isin(
                    candidates,
                    relevant,
                ).sum()
                / len(relevant)
            )

            for k in K_VALUES:
                metric_sums[
                    k
                ] += evaluate_user(
                    recommended=(
                        recommendations
                    ),
                    relevant=(
                        relevant
                    ),
                    k=k,
                )

        completed = min(
            start
            + EVAL_BATCH_SIZE,
            len(users),
        )

        if (
            completed % 10_000
            < EVAL_BATCH_SIZE
        ):
            print(
                f"Evaluated "
                f"{completed:,}/"
                f"{len(users):,} users"
            )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    candidate_recall = (
        candidate_recall_sum
        / len(users)
    )

    rows = []

    for k in K_VALUES:
        (
            recall,
            ndcg,
            mrr,
            map_k,
            hit_rate,
        ) = (
            metric_sums[
                k
            ]
            / len(users)
        )

        rows.append(
            {
                "Model": (
                    "TwoStageLearned"
                ),
                "K": k,
                "EvaluatedUsers": (
                    len(users)
                ),
                "Recall": (
                    recall
                ),
                "NDCG": (
                    ndcg
                ),
                "MRR": (
                    mrr
                ),
                "MAP": (
                    map_k
                ),
                "HitRate": (
                    hit_rate
                ),
                "CandidateRecall@100": (
                    candidate_recall
                ),
                "ALSWeight": (
                    np.nan
                ),
                "PopularityWeight": (
                    np.nan
                ),
            }
        )

    evaluation = pd.DataFrame(
        rows
    )

    evaluation.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    latency = {
        "evaluated_users": int(
            len(users)
        ),
        "retrieval_seconds": (
            retrieval_seconds
        ),
        "retrieval_ms_per_user": (
            1000
            * retrieval_seconds
            / len(users)
        ),
        "feature_and_rerank_seconds": (
            rerank_seconds
        ),
        "feature_and_rerank_ms_per_user": (
            1000
            * rerank_seconds
            / len(users)
        ),
        "end_to_end_seconds": (
            total_seconds
        ),
        "end_to_end_ms_per_user": (
            1000
            * total_seconds
            / len(users)
        ),
    }

    return (
        evaluation,
        latency,
    )


# ============================================================
# Four-model comparison
# ============================================================

def build_comparison(
    learned_evaluation,
):
    """
    Combine the Step 17 results with the learned reranker.

    Expected existing models:
        Popularity
        ALS
        TwoStageHeuristic
    """

    prior_path = (
        DATA_DIR
        / "two_stage_model_comparison.csv"
    )

    if not prior_path.exists():
        raise FileNotFoundError(
            "Run Step 17 before Step 18. "
            "two_stage_model_comparison.csv "
            "was not found."
        )

    prior = pd.read_csv(
        prior_path
    )

    required_models = {
        "Popularity",
        "ALS",
        "TwoStageHeuristic",
    }

    found_models = set(
        prior[
            "Model"
        ].unique()
    )

    if not required_models.issubset(
        found_models
    ):
        raise ValueError(
            "Step 17 comparison is missing "
            "a required model."
        )

    prior = prior[
        prior[
            "Model"
        ].isin(
            required_models
        )
        & prior[
            "K"
        ].isin(
            K_VALUES
        )
    ].copy()

    expected_rows = (
        len(required_models)
        * len(K_VALUES)
    )

    if len(prior) != expected_rows:
        raise ValueError(
            "Unexpected number of Step 17 "
            "comparison rows."
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
            prior[
                column
            ] = np.nan

        if column not in learned_evaluation:
            learned_evaluation[
                column
            ] = np.nan

    comparison = pd.concat(
        [
            prior[
                columns
            ],
            learned_evaluation[
                columns
            ],
        ],
        ignore_index=True,
    )

    model_order = {
        "Popularity": 0,
        "ALS": 1,
        "TwoStageHeuristic": 2,
        "TwoStageLearned": 3,
    }

    comparison[
        "_model_order"
    ] = comparison[
        "Model"
    ].map(
        model_order
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
            columns=(
                "_model_order"
            )
        )
        .reset_index(
            drop=True
        )
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    return comparison


# ============================================================
# Main
# ============================================================

def main():
    set_seed(
        SEED
    )

    check_metrics()

    device = choose_device()

    print(
        "PyTorch device:",
        device,
    )

    # --------------------------------------------------------
    # Load train-only artifacts
    # --------------------------------------------------------

    train_matrix = load_npz(
        DATA_DIR
        / "train_interactions.npz"
    ).tocsr()

    user_factors = np.load(
        DATA_DIR
        / "als_user_factors.npy",
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
        DATA_DIR
        / "popularity_scores.npy"
    ).astype(
        np.float32,
        copy=False,
    )

    user_idx_map = load_json(
        DATA_DIR
        / "user_idx_map.json"
    )

    item_idx_map = load_json(
        DATA_DIR
        / "item_idx_map.json"
    )

    idx_user_map = load_json(
        DATA_DIR
        / "idx_user_map.json"
    )

    idx_item_map = load_json(
        DATA_DIR
        / "idx_item_map.json"
    )

    if (
        user_factors.shape[0]
        != train_matrix.shape[0]
    ):
        raise ValueError(
            "User factors do not match "
            "the train matrix."
        )

    if (
        item_factors.shape[0]
        != train_matrix.shape[1]
    ):
        raise ValueError(
            "Item factors do not match "
            "the train matrix."
        )

    if (
        len(popularity)
        != train_matrix.shape[1]
    ):
        raise ValueError(
            "Popularity scores do not match "
            "the train matrix."
        )

    (
        item_categories,
        item_subcategories,
        title_map,
        num_categories,
        num_subcategories,
    ) = load_item_metadata(
        item_idx_map=(
            item_idx_map
        ),
        num_items=(
            train_matrix.shape[1]
        ),
    )

    # --------------------------------------------------------
    # Build/reuse sampled ranker training features
    # --------------------------------------------------------

    (
        feature_summary,
        mean,
        std,
    ) = build_feature_parts(
        user_idx_map=(
            user_idx_map
        ),
        item_idx_map=(
            item_idx_map
        ),
        user_factors=(
            user_factors
        ),
        item_factors=(
            item_factors
        ),
        popularity=(
            popularity
        ),
        item_categories=(
            item_categories
        ),
        item_subcategories=(
            item_subcategories
        ),
    )

    print(
        "\nFeature mean:"
    )

    print(
        dict(
            zip(
                FEATURE_NAMES,
                mean,
            )
        )
    )

    print(
        "\nFeature standard deviation:"
    )

    print(
        dict(
            zip(
                FEATURE_NAMES,
                std,
            )
        )
    )

    # --------------------------------------------------------
    # Train the learned reranker
    # --------------------------------------------------------

    training_seconds = train_ranker(
        mean=mean,
        std=std,
        device=device,
    )

    (
        model,
        checkpoint_mean,
        checkpoint_std,
        checkpoint,
    ) = load_ranker(
        device
    )

    print(
        "Best epoch:",
        checkpoint[
            "best_epoch"
        ],
    )

    print(
        "Best validation loss:",
        checkpoint[
            "best_validation_loss"
        ],
    )

    # --------------------------------------------------------
    # Build exact FAISS IndexFlatIP
    # --------------------------------------------------------

    index_start = (
        time.perf_counter()
    )

    index = faiss.IndexFlatIP(
        item_factors.shape[1]
    )

    index.add(
        item_factors
    )

    index_seconds = (
        time.perf_counter()
        - index_start
    )

    # --------------------------------------------------------
    # Save one sample candidate list and learned top-10
    # --------------------------------------------------------

    save_sample(
        index=index,
        model=model,
        mean=checkpoint_mean,
        std=checkpoint_std,
        train_matrix=train_matrix,
        user_factors=user_factors,
        item_factors=item_factors,
        popularity=popularity,
        item_categories=(
            item_categories
        ),
        item_subcategories=(
            item_subcategories
        ),
        num_categories=(
            num_categories
        ),
        num_subcategories=(
            num_subcategories
        ),
        idx_user_map=(
            idx_user_map
        ),
        idx_item_map=(
            idx_item_map
        ),
        title_map=(
            title_map
        ),
        device=device,
    )

    # --------------------------------------------------------
    # Load official dev labels only after training is complete
    # --------------------------------------------------------

    dev_matrix = load_npz(
        DATA_DIR
        / "dev_interactions.npz"
    ).tocsr()

    if (
        train_matrix.shape
        != dev_matrix.shape
    ):
        raise ValueError(
            "Train and dev matrix shapes "
            "do not match."
        )

    (
        evaluation,
        latency,
    ) = evaluate_learned(
        index=index,
        model=model,
        mean=checkpoint_mean,
        std=checkpoint_std,
        train_matrix=train_matrix,
        dev_matrix=dev_matrix,
        user_factors=user_factors,
        item_factors=item_factors,
        popularity=popularity,
        item_categories=(
            item_categories
        ),
        item_subcategories=(
            item_subcategories
        ),
        num_categories=(
            num_categories
        ),
        num_subcategories=(
            num_subcategories
        ),
        device=device,
    )

    latency.update(
        {
            "feature_build_seconds": (
                feature_summary.get(
                    "feature_build_seconds"
                )
            ),
            "ranker_training_seconds": (
                training_seconds
            ),
            "faiss_index_build_seconds": (
                index_seconds
            ),
        }
    )

    with open(
        LATENCY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            latency,
            file,
            indent=2,
        )

    comparison = build_comparison(
        evaluation
    )

    print(
        "\nLearned reranker evaluation:"
    )

    print(
        evaluation.round(
            6
        ).to_string(
            index=False
        )
    )

    print(
        "\nFour-model comparison:"
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