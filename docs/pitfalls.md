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





