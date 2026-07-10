# ============================================================
# 14 - Train Implicit ALS on Full MIND
#
# Input:
#   data/processed/mindlarge/train_interactions.npz
#
# Outputs:
#   data/processed/mindlarge/als_model.npz
#   data/processed/mindlarge/als_user_factors.npy
#   data/processed/mindlarge/als_item_factors.npy
# ============================================================

import time
from pathlib import Path

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mindlarge"

FACTORS = 64
REGULARIZATION = 0.1
ALPHA = 40.0
ITERATIONS = 15
RANDOM_STATE = 42


def main():

    train_matrix = load_npz(
        DATA_DIR / "train_interactions.npz"
    ).tocsr().astype(np.float32)

    print("Train matrix shape:", train_matrix.shape)
    print("Train matrix nnz:", f"{train_matrix.nnz:,}")
    print("Matrix dtype:", train_matrix.dtype)

    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        alpha=ALPHA,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        use_gpu=False,
    )

    print("\nTraining Full MIND ALS...")
    start_time = time.perf_counter()

    model.fit(
        train_matrix,
        show_progress=True,
    )

    training_time = time.perf_counter() - start_time

    model.save(
        str(DATA_DIR / "als_model.npz")
    )

    np.save(
        DATA_DIR / "als_user_factors.npy",
        model.user_factors,
    )

    np.save(
        DATA_DIR / "als_item_factors.npy",
        model.item_factors,
    )

    print("\nALS training complete.")
    print("User factors shape:", model.user_factors.shape)
    print("Item factors shape:", model.item_factors.shape)
    print("Factor dtype:", model.user_factors.dtype)
    print("Training time:", f"{training_time:.2f} seconds")

    print("\nSaved:")
    print(DATA_DIR / "als_model.npz")
    print(DATA_DIR / "als_user_factors.npy")
    print(DATA_DIR / "als_item_factors.npy")


if __name__ == "__main__":
    main()