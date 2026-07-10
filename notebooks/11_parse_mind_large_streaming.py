# ============================================================
# 11 - Full MIND Streaming Behavior Processing
#
# Goal:
# Stream the Full MIND behavior files without loading the entire
# dataset into memory, and save positive interactions as multiple
# Parquet part files.
#
# Train positive interactions:
#   1. Every news item in the user's history field.
#   2. Every impression item whose click label is 1.
#
# Dev validation interactions:
#   1. Only impression items whose click label is 1.
#   2. Dev history is intentionally excluded because it describes
#      past behavior rather than future validation targets.
#
# Important:
# This script does not globally deduplicate user-item pairs.
# Maintaining a global Python set for tens of millions of pairs
# would defeat the purpose of memory-safe streaming.
#
# Duplicate coordinates will be consolidated and converted to
# binary values when the Full MIND sparse matrix is constructed
# in a later script.
#
# Inputs:
#   data/raw/MINDlarge_train/behaviors.tsv
#   data/raw/MINDlarge_dev/behaviors.tsv
#
# Outputs:
#   data/processed/mindlarge/train_positive_interactions/
#       part-00000.parquet
#       part-00001.parquet
#       ...
#
#   data/processed/mindlarge/dev_clicked_impressions/
#       part-00000.parquet
#       part-00001.parquet
#       ...
#
#   data/processed/mindlarge/11_streaming_parse_summary.json
# ============================================================

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

# This script is stored inside:
#
#     project_root/notebooks/
#
# parents[1] therefore points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_BEHAVIORS_PATH = (
    RAW_DATA_DIR
    / "MINDlarge_train"
    / "behaviors.tsv"
)

DEV_BEHAVIORS_PATH = (
    RAW_DATA_DIR
    / "MINDlarge_dev"
    / "behaviors.tsv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mindlarge"
)

TRAIN_OUTPUT_DIR = (
    PROCESSED_DIR
    / "train_positive_interactions"
)

DEV_OUTPUT_DIR = (
    PROCESSED_DIR
    / "dev_clicked_impressions"
)

SUMMARY_PATH = (
    PROCESSED_DIR
    / "11_streaming_parse_summary.json"
)


# ------------------------------------------------------------
# Processing settings
# ------------------------------------------------------------

# The script keeps at most approximately this many interaction
# rows in memory before writing one Parquet part file.
#
# A smaller value uses less memory but creates more files.
# A larger value creates fewer files but uses more memory.
BUFFER_ROWS = 500_000

# Print progress after this many raw behavior rows.
PROGRESS_EVERY = 100_000

PARQUET_COMPRESSION = "snappy"


# ------------------------------------------------------------
# Output-directory preparation
# ------------------------------------------------------------

def reset_output_directory(output_dir):
    """
    Remove an old output directory and create a clean directory.

    This makes rerunning the script deterministic and prevents old
    Parquet part files from being mixed with newly generated files.
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# Write one buffered Parquet part
# ------------------------------------------------------------

def write_parquet_part(
    user_buffer,
    item_buffer,
    source_buffer,
    output_dir,
    part_number,
):
    """
    Write buffered positive interactions to one Parquet part.

    Each output row contains:

        user_id
        item_id
        click
        source

    Every row in this script is a positive interaction, so click is
    always equal to 1.
    """

    if not user_buffer:
        return 0

    number_of_rows = len(user_buffer)

    interaction_df = pd.DataFrame(
        {
            "user_id": user_buffer,
            "item_id": item_buffer,

            # int8 is sufficient because click is binary.
            "click": np.ones(
                number_of_rows,
                dtype=np.int8,
            ),

            "source": source_buffer,
        }
    )

    # Only two source values are used:
    #
    #     history
    #     impression
    #
    # Category dtype allows Parquet to store repeated values more
    # compactly.
    interaction_df["source"] = (
        interaction_df["source"].astype("category")
    )

    output_path = (
        output_dir
        / f"part-{part_number:05d}.parquet"
    )

    interaction_df.to_parquet(
        output_path,
        index=False,
        compression=PARQUET_COMPRESSION,
    )

    print(
        f"  Saved {output_path.name}: "
        f"{number_of_rows:,} rows",
        flush=True,
    )

    # Clear the original lists so that their memory can be reused.
    user_buffer.clear()
    item_buffer.clear()
    source_buffer.clear()

    return number_of_rows


# ------------------------------------------------------------
# Parse one MIND behaviors.tsv file
# ------------------------------------------------------------

def parse_behavior_file(
    input_path,
    output_dir,
    split_name,
    include_history,
):
    """
    Stream one MIND behavior file and save positive interactions.

    Parameters
    ----------
    input_path:
        Path to behaviors.tsv.

    output_dir:
        Directory receiving Parquet part files.

    split_name:
        Human-readable split name used in progress messages.

    include_history:
        True for train:
            save history clicks and clicked impressions.

        False for dev:
            save clicked impressions only.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    reset_output_directory(output_dir)

    start_time = time.perf_counter()

    # Buffered output columns.
    user_buffer = []
    item_buffer = []
    source_buffer = []

    # Diagnostic counters.
    behavior_rows = 0
    malformed_behavior_rows = 0
    malformed_impression_tokens = 0

    history_interactions = 0
    clicked_impression_interactions = 0

    rows_written = 0
    part_number = 0

    print(
        f"\nProcessing {split_name}:",
        input_path,
        flush=True,
    )

    # MIND files are UTF-8 tab-separated files without a header.
    #
    # Each behavior row contains:
    #
    # impression_id
    # user_id
    # time
    # history
    # impressions
    with input_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as behavior_file:

        for line in behavior_file:

            behavior_rows += 1

            # Remove the line ending but preserve empty TSV fields.
            line = line.rstrip("\r\n")

            # maxsplit=4 guarantees that at most five fields are
            # created.
            fields = line.split("\t", 4)

            if len(fields) != 5:
                malformed_behavior_rows += 1
                continue

            (
                _impression_id,
                user_id,
                _timestamp,
                history,
                impressions,
            ) = fields

            if not user_id:
                malformed_behavior_rows += 1
                continue

            # ------------------------------------------------
            # Train history clicks
            # ------------------------------------------------

            # The history field contains previously clicked news:
            #
            #     N123 N456 N789
            #
            # Dev history is deliberately excluded because the dev
            # output will later be used as validation ground truth.
            if include_history and history:

                history_items = history.split()

                for item_id in history_items:
                    user_buffer.append(user_id)
                    item_buffer.append(item_id)
                    source_buffer.append("history")

                history_interactions += len(
                    history_items
                )

            # ------------------------------------------------
            # Clicked impression items
            # ------------------------------------------------

            # Each impression token has the format:
            #
            #     N12345-1
            #     N67890-0
            #
            # We only save label=1 as a positive interaction.
            if impressions:

                for token in impressions.split():

                    try:
                        item_id, click_label = token.rsplit(
                            "-",
                            1,
                        )
                    except ValueError:
                        malformed_impression_tokens += 1
                        continue

                    if not item_id:
                        malformed_impression_tokens += 1
                        continue

                    if click_label == "1":
                        user_buffer.append(user_id)
                        item_buffer.append(item_id)
                        source_buffer.append("impression")

                        clicked_impression_interactions += 1

                    elif click_label != "0":
                        # Valid MIND impression labels should be
                        # either 0 or 1.
                        malformed_impression_tokens += 1

            # ------------------------------------------------
            # Flush the buffer
            # ------------------------------------------------

            # Check after finishing one raw behavior row. The buffer
            # may exceed BUFFER_ROWS slightly if one row contains
            # many history items, which is harmless.
            if len(user_buffer) >= BUFFER_ROWS:

                rows_written += write_parquet_part(
                    user_buffer=user_buffer,
                    item_buffer=item_buffer,
                    source_buffer=source_buffer,
                    output_dir=output_dir,
                    part_number=part_number,
                )

                part_number += 1

            # ------------------------------------------------
            # Progress output
            # ------------------------------------------------

            if behavior_rows % PROGRESS_EVERY == 0:

                elapsed_seconds = (
                    time.perf_counter()
                    - start_time
                )

                extracted_rows = (
                    history_interactions
                    + clicked_impression_interactions
                )

                print(
                    f"[{split_name}] "
                    f"behavior rows: {behavior_rows:,} | "
                    f"positive rows extracted: "
                    f"{extracted_rows:,} | "
                    f"elapsed: {elapsed_seconds:.1f} seconds",
                    flush=True,
                )

    # Write any interactions remaining after the final input line.
    if user_buffer:

        rows_written += write_parquet_part(
            user_buffer=user_buffer,
            item_buffer=item_buffer,
            source_buffer=source_buffer,
            output_dir=output_dir,
            part_number=part_number,
        )

        part_number += 1

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    expected_rows = (
        history_interactions
        + clicked_impression_interactions
    )

    if rows_written != expected_rows:
        raise ValueError(
            f"{split_name}: rows written ({rows_written}) "
            f"do not match extracted rows ({expected_rows})."
        )

    result = {
        "split": split_name,
        "input_path": str(input_path),
        "behavior_rows": behavior_rows,
        "history_interactions": history_interactions,
        "clicked_impression_interactions": (
            clicked_impression_interactions
        ),
        "total_positive_rows_written": rows_written,
        "parquet_part_files": part_number,
        "malformed_behavior_rows": (
            malformed_behavior_rows
        ),
        "malformed_impression_tokens": (
            malformed_impression_tokens
        ),
        "elapsed_seconds": elapsed_seconds,
        "output_directory": str(output_dir),
    }

    print(
        f"\nFinished {split_name}:",
        flush=True,
    )

    print(
        f"  Raw behavior rows: "
        f"{behavior_rows:,}",
        flush=True,
    )

    print(
        f"  History interactions: "
        f"{history_interactions:,}",
        flush=True,
    )

    print(
        f"  Clicked impression interactions: "
        f"{clicked_impression_interactions:,}",
        flush=True,
    )

    print(
        f"  Total positive rows written: "
        f"{rows_written:,}",
        flush=True,
    )

    print(
        f"  Parquet part files: "
        f"{part_number:,}",
        flush=True,
    )

    print(
        f"  Malformed behavior rows: "
        f"{malformed_behavior_rows:,}",
        flush=True,
    )

    print(
        f"  Malformed impression tokens: "
        f"{malformed_impression_tokens:,}",
        flush=True,
    )

    print(
        f"  Processing time: "
        f"{elapsed_seconds:.2f} seconds",
        flush=True,
    )

    return result


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Full MIND streaming processing",
        flush=True,
    )

    print(
        "Train input:",
        TRAIN_BEHAVIORS_PATH,
        flush=True,
    )

    print(
        "Dev input:",
        DEV_BEHAVIORS_PATH,
        flush=True,
    )

    print(
        "Buffer rows:",
        f"{BUFFER_ROWS:,}",
        flush=True,
    )

    # Train contains both historical clicks and clicked
    # impression items.
    train_result = parse_behavior_file(
        input_path=TRAIN_BEHAVIORS_PATH,
        output_dir=TRAIN_OUTPUT_DIR,
        split_name="train",
        include_history=True,
    )

    # Dev contains clicked impression items only.
    dev_result = parse_behavior_file(
        input_path=DEV_BEHAVIORS_PATH,
        output_dir=DEV_OUTPUT_DIR,
        split_name="dev",
        include_history=False,
    )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "buffer_rows": BUFFER_ROWS,
        "progress_every": PROGRESS_EVERY,
        "parquet_compression": PARQUET_COMPRESSION,
        "global_deduplication_performed": False,
        "deduplication_note": (
            "Duplicate user-item pairs are intentionally retained "
            "during streaming and will be consolidated when the "
            "binary sparse interaction matrix is constructed."
        ),
        "train": train_result,
        "dev": dev_result,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
        )

    print(
        "\nStreaming processing complete.",
        flush=True,
    )

    print(
        "Summary saved:",
        SUMMARY_PATH,
        flush=True,
    )

    print(
        "Train output:",
        TRAIN_OUTPUT_DIR,
        flush=True,
    )

    print(
        "Dev output:",
        DEV_OUTPUT_DIR,
        flush=True,
    )


if __name__ == "__main__":
    main()