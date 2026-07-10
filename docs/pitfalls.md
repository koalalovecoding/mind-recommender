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





## 2026-07-10 Pitfall: Full MIND Step 18 failed because of memory pressure and a macOS OpenMP conflict

**Where.**
Files:

```text
notebooks/18_mind_large_learned_reranker.py
notebooks/18a_mind_large_faiss_candidates.py
notebooks/18b_mind_large_learned_reranker_evaluation.py
```

**Problem.**
The first attempts to run the Full MIND learned-reranker pipeline failed twice for two different system-level reasons.

---

### Failure 1: In-memory feature construction caused a segmentation fault

The shortened version of Step 18 constructed all ranker samples in memory.

The generated sample counts were:

```text
Ranker train samples:      14,536,251
Ranker validation samples:  1,615,970
```

Immediately after these arrays were created, the process terminated with:

```text
Segmentation fault: 11
```

**Cause.**
The shortened implementation stored all feature arrays, label arrays, intermediate chunk arrays, standardized copies, sparse matrices, mappings, and ALS factors in the same process.

It also created a PyTorch `DataLoader` with:

```python
shuffle=True
```

For more than 14 million training samples, shuffling requires a very large random index permutation in addition to the existing feature arrays.

On Apple Silicon, the CPU and MPS backend share unified system memory. The combination of large in-memory arrays, temporary copies, PyTorch tensors, random sampling indices, and MPS allocation created an excessive memory peak.

This was not a model-definition error. The failure was caused by using an in-memory data pipeline that was inappropriate for Full MIND scale.

**Fix.**
The memory-safe implementation was restored.

It:

```text
1. Streams the official train behavior file.
2. Writes candidate-level features to NumPy part files.
3. Stores 500,000 samples per part.
4. Loads only one part at a time during training.
5. Yields already-vectorized mini-batches through an IterableDataset.
```

The final feature storage contained:

```text
30 train parts
4 validation parts
34 total parts
```

This allowed the complete PyTorch training process to finish without exhausting memory.

---

### Failure 2: PyTorch and FAISS loaded conflicting OpenMP runtimes

After the memory-safe pipeline successfully generated the feature parts and trained the MLP, the same process attempted to initialize FAISS.

The training completed successfully:

```text
Best epoch: 8
Best validation loss: 0.3389157276458985
```

However, the process then failed with:

```text
OMP: Error #15: Initializing libomp.dylib,
but found libomp.dylib already initialized.

Abort trap: 6
```

**Initial diagnostic.**
The terminal originally showed both environments:

```text
(.venv) (base)
```

Conda `base` was deactivated, leaving only:

```text
(.venv)
```

The active Python executable was verified as:

```text
/Users/hl/Desktop/PythonProject1/mind/.venv/bin/python
```

A minimal test that imported and executed both PyTorch/MPS and FAISS still reproduced the same OpenMP error.

This ruled out the nested Conda environment as the primary cause.

**Root cause.**
On the current macOS Apple Silicon environment, the installed PyTorch and FAISS packages each initialized an OpenMP runtime.

The original Step 18 script imported both libraries:

```python
import torch
import faiss
```

When the second OpenMP runtime was initialized in the same Python process, the runtime aborted the program.

The environment variable:

```bash
KMP_DUPLICATE_LIB_OK=TRUE
```

was not used as the final solution because it only suppresses the duplicate-runtime check. It may allow the process to continue while still risking crashes, degraded performance, or silently incorrect results.

**Correct fix.**
Step 18 was split into two independent processes:

```text
18a:
FAISS candidate retrieval only
No PyTorch import

18b:
PyTorch learned reranking and evaluation only
No FAISS import
```

The two stages communicate through saved NumPy files containing:

```text
evaluated user indices
top-100 candidate item indices
sample-user candidates
sample-user FAISS scores
```

Because FAISS and PyTorch now run in different operating-system processes, their OpenMP runtimes are never loaded into the same process.

---

### Result

The separated pipeline completed successfully:

```text
18a:
FAISS top-100 candidate retrieval completed for 205,536 users.

18b:
PyTorch feature construction, learned reranking, metric evaluation,
sample recommendation output, model comparison, and latency reporting
all completed successfully.
```

**Lesson.**
For large recommendation workloads, simplifying code by removing streaming and partitioned storage can make the implementation unsuitable for the actual dataset scale.

In addition, native numerical libraries may conflict even when the Python virtual environment is correctly configured. When two required libraries cannot safely coexist in one process, process-level separation is safer than forcing duplicate runtimes to load.

**Interview note.**
The final architecture separates candidate generation and reranking into independent executable stages. This is also consistent with production recommendation systems, where retrieval and ranking are often deployed as separate services or jobs.



