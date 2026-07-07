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


