# Pitfalls


## 2026-07-05 Pitfall: `git add` path depends on the current directory

**Problem.**  
Running this command failed:

```bash
git add notebooks/02_mind_data_processing.ipynb
```

### Error:

```text
fatal: pathspec 'notebooks/02_mind_data_processing.ipynb' did not match any files
```

### Cause:

The terminal was inside:

```text
data/raw/
```

So Git interpreted the path relative to `data/raw/`, not relative to the project root.

### Fix:

Move back to the project root:

```bash
cd ../..
```

Then run:

```bash
git add notebooks/02_mind_data_processing.ipynb
```

### Lesson:

Run project-level Git commands from the repository root, or use the correct relative path.

### Useful check:

```bash
pwd
ls
find . -name "*.ipynb"
```

### Interview note:

This is a basic reproducibility and project-organization issue. The fix was to understand paths relative to the current working directory.

---


## 2026-07-07 Pitfall: Avoid large-scale MIND processing in notebooks

**Problem.**  
Using `.ipynb` notebooks to process the expanded MIND-small interaction tables can cause the notebook kernel or IDE to crash, especially after parsing impressions and user histories into millions of rows.

**Why it happens.**  
The processed interaction tables are large. Notebook environments keep many intermediate variables in memory and can become unstable when repeatedly running heavy data-processing cells.

**Fix.**  
Use standalone `.py` scripts for large data-processing steps, especially parsing, merging, and saving parquet files. Use notebooks only for inspection, small samples, explanation, and visualization.

**Rule.**  
Heavy processing goes into `.py`; notebooks are for exploration and reporting.


## 2026-07-07 Pitfall: Exclude large data files from PyCharm indexing

**Problem.**  
Keeping the full `data/` directory visible to PyCharm can make the IDE slow or unresponsive because it may try to index large raw and processed files.

**Why it happens.**  
MIND-small raw files and processed parquet files can be large. PyCharm indexing these files is unnecessary and can consume a lot of memory and CPU.

**Fix.**  
Mark the `data/` directory as excluded in PyCharm. The files should still remain in the project folder, but PyCharm should not index them. Load them explicitly in code only when needed.

**Rule.**  
Exclude `data/` from IDE indexing; access data files through paths in scripts.




## 2026-07-07 Pitfall: Do not use dev history rows when building the dev sparse matrix

**Where.**
File: `notebooks/06_user_item_interaction_matrix`
Step: Phase 2 Part B, Step 8 — build `dev_interactions.npz`.

**Problem.**
In Step 8, the dev sparse matrix is built from `dev_with_news.parquet`. A tempting but incorrect first version is:

```python
dev_pos = dev_df[dev_df["click"] == 1][["user_id", "item_id"]].drop_duplicates()
```

This selects all positive rows in the dev interaction table. However, the dev table contains both `source = "history"` rows and `source = "impression"` rows. As a result, this code would include dev history clicks in `dev_interactions.npz`.

**Why this is a pitfall.**
The dev sparse matrix is meant to be the validation ground-truth matrix for later ranking evaluation. It should contain the clicked items from dev impressions, because those are the target interactions the recommender is expected to recover. Dev history clicks are past clicks used to describe the user's previous behavior; they should not be mixed into the validation target.

**Correct implementation.**
When building `dev_interactions.npz`, use only clicked dev impressions:

```python
dev_pos = dev_df[
    (dev_df["click"] == 1) &
    (dev_df["source"] == "impression")
][["user_id", "item_id"]].drop_duplicates()
```

Then apply the warm-start filter:

```python
dev_pos = dev_pos[
    dev_pos["user_id"].isin(user_idx_map)
    & dev_pos["item_id"].isin(item_idx_map)
]
```

**Correct interpretation.**
The train sparse matrix can use positive train interactions from both `source = "history"` and `source = "impression"`, because both are training signals. The dev sparse matrix should only use `source = "impression"` with `click = 1`, because it is used as validation ground truth, not as additional user history.

**Takeaway.**
For Step 8, train positives and dev validation positives are not selected in exactly the same way: train uses all positive train clicks, while dev uses only clicked dev impressions.



