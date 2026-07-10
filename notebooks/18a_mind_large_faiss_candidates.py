# ============================================================
# 18a - Full MIND FAISS Candidate Retrieval
#
# This process imports FAISS but does not import PyTorch.
# It retrieves and saves top-100 unseen candidates for every
# evaluable dev user, avoiding the macOS FAISS/PyTorch OpenMP clash.
# ============================================================

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"

CANDIDATE_K = 100
BATCH_SIZE = 256
SAMPLE_USER_IDX = 0

# Keep this as None for the full experiment.
DEBUG_MAX_EVAL_USERS = None

EVAL_USERS_PATH = DATA_DIR / "learned_reranker_eval_users.npy"
CANDIDATES_PATH = DATA_DIR / "learned_reranker_faiss_candidates.npy"
SAMPLE_USER_PATH = DATA_DIR / "learned_reranker_sample_user.npy"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "learned_reranker_sample_candidates.npy"
SAMPLE_SCORES_PATH = DATA_DIR / "learned_reranker_sample_faiss_scores.npy"
SUMMARY_PATH = DATA_DIR / "learned_reranker_retrieval_summary.json"


def get_unseen_candidates(items, scores, seen_items):
    """Remove train-seen items and keep exactly 100 unique items."""

    valid = items >= 0
    items = items[valid]
    scores = scores[valid]

    unseen = ~np.isin(items, seen_items)
    items = items[unseen][:CANDIDATE_K]
    scores = scores[unseen][:CANDIDATE_K]

    if len(items) != CANDIDATE_K:
        raise ValueError("Could not retrieve 100 unseen candidates.")

    if len(np.unique(items)) != CANDIDATE_K:
        raise ValueError("Candidate items contain duplicates.")

    if np.intersect1d(items, seen_items).size:
        raise ValueError("A train-seen item remains in the candidates.")

    return (
        items.astype(np.int32, copy=False),
        scores.astype(np.float32, copy=False),
    )


def retrieve_one_user(index, user_idx, train_matrix, user_factors):
    """Retrieve one user's top-100 unseen candidates."""

    user_vector = np.ascontiguousarray(
        user_factors[user_idx],
        dtype=np.float32,
    )

    seen_items = train_matrix[user_idx].indices

    search_k = min(
        train_matrix.shape[1],
        CANDIDATE_K + len(seen_items),
    )

    scores, items = index.search(
        user_vector[None, :],
        search_k,
    )

    return get_unseen_candidates(
        items[0],
        scores[0],
        seen_items,
    )


def main():
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

    if train_matrix.shape != dev_matrix.shape:
        raise ValueError("Train and dev matrix shapes do not match.")

    if user_factors.shape[0] != train_matrix.shape[0]:
        raise ValueError("User factors do not match train users.")

    if item_factors.shape[0] != train_matrix.shape[1]:
        raise ValueError("Item factors do not match train items.")

    users = np.flatnonzero(
        np.diff(dev_matrix.indptr) > 0
    ).astype(np.int32)

    if DEBUG_MAX_EVAL_USERS is not None:
        users = users[:DEBUG_MAX_EVAL_USERS]

    if len(users) == 0:
        raise ValueError("No evaluable dev users were found.")

    print("Evaluated users:", f"{len(users):,}")

    index_start = time.perf_counter()

    index = faiss.IndexFlatIP(item_factors.shape[1])
    index.add(item_factors)

    index_seconds = time.perf_counter() - index_start

    print(
        "FAISS index built:",
        f"{index.ntotal:,} items in {index_seconds:.2f}s",
    )

    # Save the sample user's candidates separately because user 0
    # may not be among the evaluable dev users.
    sample_user_idx = SAMPLE_USER_IDX

    if train_matrix[sample_user_idx].nnz == 0:
        sample_user_idx = int(
            np.flatnonzero(
                np.diff(train_matrix.indptr) > 0
            )[0]
        )

    sample_candidates, sample_scores = retrieve_one_user(
        index,
        sample_user_idx,
        train_matrix,
        user_factors,
    )

    np.save(
        SAMPLE_USER_PATH,
        np.asarray(sample_user_idx, dtype=np.int32),
        allow_pickle=False,
    )

    np.save(
        SAMPLE_CANDIDATES_PATH,
        sample_candidates,
        allow_pickle=False,
    )

    np.save(
        SAMPLE_SCORES_PATH,
        sample_scores,
        allow_pickle=False,
    )

    # Write to a temporary .npy file and rename only after the
    # complete retrieval pass succeeds.
    temporary_path = CANDIDATES_PATH.with_name(
        CANDIDATES_PATH.stem + ".tmp.npy"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    candidate_memmap = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(users), CANDIDATE_K),
    )

    candidate_recall_sum = 0.0
    retrieval_start = time.perf_counter()

    for start in range(0, len(users), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(users))

        batch_users = users[start:end]
        batch_train = train_matrix[batch_users]

        batch_vectors = np.ascontiguousarray(
            user_factors[batch_users],
            dtype=np.float32,
        )

        max_seen = int(
            np.diff(batch_train.indptr).max()
        )

        search_k = min(
            train_matrix.shape[1],
            CANDIDATE_K + max_seen,
        )

        batch_scores, batch_items = index.search(
            batch_vectors,
            search_k,
        )

        for row, user_idx in enumerate(batch_users):
            seen_items = batch_train[row].indices

            candidates, _ = get_unseen_candidates(
                batch_items[row],
                batch_scores[row],
                seen_items,
            )

            candidate_memmap[start + row] = candidates

            relevant = dev_matrix[user_idx].indices

            candidate_recall_sum += (
                np.isin(candidates, relevant).sum()
                / len(relevant)
            )

        candidate_memmap.flush()

        if end % 10_000 < BATCH_SIZE or end == len(users):
            elapsed = time.perf_counter() - retrieval_start

            print(
                f"Retrieved {end:,}/{len(users):,} users "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    retrieval_seconds = time.perf_counter() - retrieval_start

    del candidate_memmap
    os.replace(temporary_path, CANDIDATES_PATH)

    np.save(
        EVAL_USERS_PATH,
        users,
        allow_pickle=False,
    )

    summary = {
        "evaluated_users": int(len(users)),
        "candidate_k": CANDIDATE_K,
        "candidate_dtype": "int32",
        "candidate_recall_at_100": (
            candidate_recall_sum / len(users)
        ),
        "faiss_index_build_seconds": index_seconds,
        "retrieval_seconds": retrieval_seconds,
        "retrieval_ms_per_user": (
            1000 * retrieval_seconds / len(users)
        ),
        "eval_users_path": str(EVAL_USERS_PATH),
        "candidates_path": str(CANDIDATES_PATH),
        "sample_user_idx": int(sample_user_idx),
        "sample_candidates_path": str(SAMPLE_CANDIDATES_PATH),
        "sample_scores_path": str(SAMPLE_SCORES_PATH),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nCandidate retrieval complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
