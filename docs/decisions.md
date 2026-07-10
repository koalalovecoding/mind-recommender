# Decision Log

## 2026-07-01   Get started -use venv 
**Decision.**: Use venv to create an isolated Python environment for this project.

**Reason.**: Keeps this project's dependencies separate from the global / conda base environment, 
so package versions across projects can't collide.
venv is lightweight and pairs cleanly with pip + requirements.txt for
reproducibility. Chose venv over conda create because the project
dependencies are pip-installable and I want requirements.txt to be the
single source of truth.

**Scope.**: this project 

## 2026-07-05 Use MIND-small for Phase 2

**Decision.**  
Use Microsoft MIND-small as the real-world dataset for Phase 2.

**Reason.**  
MIND is a news recommendation dataset with user click histories, impression logs, news titles, abstracts, 
categories, and metadata. It is more aligned with a real industrial recommendation scenario than MovieLens. 
I use MIND-small first because it is large enough to be realistic but still manageable for local data processing 
and classical recommendation baselines.

**Note.**  
This choice keeps the project connected to implicit feedback recommendation, exposure bias, and top-K ranking evaluation, 
while avoiding the engineering overhead of starting with the full MIND dataset.


## 2026-07-05 Use behaviors.tsv and news.tsv first

**Decision.**  
In the first version of Phase 2, only use `behaviors.tsv` and `news.tsv`.

**Reason.**  
`behaviors.tsv` contains user histories and impression logs, which are necessary for constructing user-item-click 
interactions. `news.tsv` contains item metadata such as category, subcategory, title, and abstract. These two 
files are sufficient for the first data processing pipeline.

**Deferred.**  
`entity_embedding.vec` and `relation_embedding.vec` are ignored for now. They can be used later for hybrid 
recommendation or knowledge-aware/content-based models.

**Note.**  
I intentionally separated the core collaborative filtering pipeline from later content-enhanced modeling. 
This keeps the phase of mind_small focused and reproducible.



## 2026-0705  Keep raw behavior logs at impression-event level

**Decision.**  
During raw data inspection (02_inspect_raw_mind_small.ipynb), do not merge rows with the same `user_id`.

**Reason.**  
Each row in `behaviors.tsv` is an impression event, not a user. The same user can appear multiple times at different 
times with different histories and different candidate news impressions. Merging users too early would lose event-level
 information such as timestamp, candidate set, and click labels.

**Later step.**  
Users will be mapped to matrix rows later when constructing the user-item sparse matrix.

## 2026-07-05 Interpret unclicked impressions carefully

**Decision.**  
Treat `Nxxxxx-0` as “exposed but not clicked,” not as a strong negative preference.

**Reason.**  
In implicit feedback recommendation, a non-click does not necessarily mean the user dislikes the item. The user may not 
have noticed the item, may not have had time, or may have ignored it for contextual reasons.

**Mathematical connection.**  
This matches the exposure-bias view:

$$R = E \odot Y$$

where exposure and latent preference are different. A zero click only tells us that the observed click did not happen; 
it does not fully reveal the latent preference.

**Note.**  The important distinction is that MIND gives exposure information. A non-clicked exposed item is not a strong 
negative preference, but it is also not the same as a completely unobserved item.


## 2026-07-06   Store processed interactions as Parquet

**Decision.**: Store processed interaction tables as `.parquet` files instead of `.csv` or `.tsv`.

**Reason.**: The parsed MIND-small interaction tables contain millions of rows, so CSV/TSV would be larger and slower to read/write. 
Parquet is more efficient for large tabular data, preserves column types better, and works cleanly with pandas through `pyarrow`.

**Scope.**: Processed data files in `data/processed/`, including `interactions_train.parquet`, `interactions_dev.parquet`, and `news.parquet`.



## 2026-07-07   Store ID mappings as JSON dictionaries

**Decision.**: Save user/item ID mappings as JSON files and load them with Python's `json.load`, rather than storing or loading them as parquet tables.

**Reason.**: The mapping objects are dictionaries, not tabular datasets. For example, `user_idx_map` maps raw `user_id` strings to integer matrix row 
indices, and `item_idx_map` maps raw `item_id` strings to integer matrix column indices. JSON preserves this key-value structure directly and makes the 
mapping easy to reuse when building sparse matrices, training models, evaluating recommendations, and converting matrix indices back to original IDs.

Parquet is better suited for DataFrame-style interaction tables such as `train_with_news.parquet` and `dev_with_news.parquet`. Although `pd.read_json` 
exists, it would read the mapping through pandas as a Series or DataFrame and then require conversion back to a dictionary, which is unnecessary for this use case.

**Scope.**: Phase 2 Part B ID mapping and sparse matrix construction.


## 2026-07-07   Use warm-start evaluation for matrix-based collaborative filtering

**Decision.**: Use warm-start evaluation as the default validation protocol for classical collaborative filtering models, and treat dev-only users or 
dev-only items as cold-start cases that should be counted separately rather than included in the standard evaluation matrix.

**Reason.**: Matrix-based collaborative filtering models such as SVD, ALS, BPR, ItemKNN, and UserKNN learn representations from the rows and columns of 
the training interaction matrix. A user that never appears in train has no learned user representation, and an item that never appears in train has no 
learned item representation. Therefore, these models cannot directly score interactions involving completely unseen users or unseen items without an additional 
cold-start strategy. Warm-start evaluation is still a valid generalization test because the model is asked to predict future user-item interactions that 
were not used for training; only the user identity and item identity are known from the training matrix.

**Cold-start strategy.**: New users or new items should be handled as a separate cold-start problem, not mixed into the ordinary matrix factorization evaluation. 
Possible strategies include popularity fallback, content-based recommendation, hybrid models, metadata features, text embeddings, or two-tower models that can 
produce representations from user/item features rather than relying only on historical interaction rows and columns.

**Scope.**: Classical collaborative filtering models, sparse train/dev matrix construction, and standard ranking evaluation.

## 2026-07-07   Remove train-seen positive pairs from dev evaluation

**Decision.**
When constructing the dev sparse matrix, remove any positive `(user_id, item_id)` pair that already appears as a positive pair in the train sparse matrix.

**Reason.**
In warm-start recommendation, the same user and the same item may appear in both train and dev, but the exact positive user-item interaction used for validation should not have already been used for training. If the same `(user_id, item_id)` positive pair appears in both train and dev, then the model has already seen that interaction as a training positive, which makes the validation target less clean.

**Scope.**
Sparse matrix construction and ranking evaluation for classical collaborative filtering models.



## 2026-07-10 Use Streaming Processing for Full MIND

**Decision.**  
Process Full MIND line by line and save positive interactions as partitioned Parquet files instead of expanding the complete dataset into one pandas DataFrame.

**Reason.**  
MIND-small was small enough to load and process as complete DataFrames. Full MIND produced more than 77 million train positive-interaction rows, so the same approach could require excessive memory.

The streaming pipeline keeps memory usage bounded by:

```text
read one behavior row
→ extract positive interactions
→ buffer about 500,000 rows
→ save one Parquet part
→ clear the buffer
```

Global `(user_id, item_id)` deduplication is delayed until sparse matrix construction.

News metadata is also kept separate instead of being copied into every interaction row.

**Scope.**  
Full MIND positive-interaction processing for user/item mappings, sparse matrix construction, and ALS training.



## 2026-07-10 Build Full MIND Sparse Matrices from Partitioned Data

**Decision.**  
Build the Full MIND sparse matrices by reading partitioned Parquet files in batches instead of loading the complete interaction table into one pandas DataFrame.

**Difference from MIND-small.**  
For MIND-small, the complete processed train/dev tables could be loaded into memory, filtered with pandas, deduplicated, and converted directly into CSR matrices.

For Full MIND, the train data contains more than 77 million positive-interaction rows across 155 Parquet files. The files are therefore processed batch by batch, 
converted into local sparse matrices, and merged while keeping all values binary.

**Unchanged semantics.**

```text
Train:
history clicks + clicked train impressions

Dev:
clicked dev impressions only
→ remove cold-start users/items
→ remove train-seen user-item pairs
```

**Reason.**  
This preserves the same user-item matrix definition as MIND-small while keeping memory usage manageable on a local machine.

## 2026-07-10 Separate Full MIND ALS Training from Recommendation Generation

**Decision.**  
Keep the Full MIND ALS training script focused on loading the train matrix, fitting ALS, and saving the trained model and latent factors.

**Difference from MIND-small.**  
The MIND-small ALS script also loaded ID mappings and news titles, generated a sample user's top-K recommendations, filtered train-seen items, and printed readable recommendation results.

The Full MIND script performs only:

```text
load train sparse matrix
→ train implicit ALS
→ save model
→ save user factors
→ save item factors
```

Sample recommendation generation and evaluation are moved to later scripts.

**Unchanged model setup.**

```text
factors:        64
regularization: 0.1
alpha:          40.0
iterations:     15
random_state:   42
```

The mathematical model and training objective are unchanged. The main difference is dataset scale:

```text
MIND-small:
50,000 users
51,282 items
1,148,447 positive pairs
approximately 5 seconds training time

Full MIND:
711,222 users
101,527 items
16,532,504 positive pairs
88.91 seconds training time
```

**Reason.**  
Separating training from recommendation generation keeps the Full MIND script short, makes each stage easier to verify, and avoids mixing model fitting with inference and evaluation logic.



## 2026-07-10 Use Log-Transformed Popularity in the Heuristic Reranker

**Decision.**
Use log-transformed item popularity rather than raw popularity when popularity is included as a feature in the MIND-small and Full MIND heuristic rerankers.

The popularity feature is calculated as:

```python
log_popularity = np.log1p(
    popularity[candidate_items]
)
```

It is then normalized within each user's candidate set:

```python
normalized_popularity = min_max(
    log_popularity
)
```

The final heuristic reranking score is:

```python
rerank_score = (
    als_weight * normalized_als_score
    + popularity_weight * normalized_popularity
)
```

**Reason.**
Raw item popularity has a highly skewed distribution. A small number of news articles may receive very large click counts, while most articles receive substantially fewer clicks. Directly combining raw popularity with ALS scores could allow a few extremely popular items to dominate the reranking score, even when the popularity weight is relatively small.

The transformation

```text
log(1 + popularity)
```

compresses large popularity values while preserving their ordering. It reduces the difference between extremely popular and moderately popular items without removing the popularity signal entirely.

`np.log1p` is used instead of `np.log` because it safely handles items with zero training clicks:

```text
log1p(0) = 0
```

**Advantages.**

```text
1. Reduces the influence of extremely popular outlier items.
2. Prevents raw click counts from dominating ALS compatibility scores.
3. Preserves the relative ordering of items by popularity.
4. Produces a more stable feature before min-max normalization.
5. Safely handles zero-popularity items.
6. Makes popularity easier to combine with ALS scores on a comparable scale.
```

**Scope.**
The log transformation is used only when popularity is combined with another feature inside the heuristic reranker.

The standalone Popularity baseline still ranks items using raw training click counts. Applying `log1p` would not change that ranking because the logarithm is strictly increasing:

```text
popularity_i > popularity_j
if and only if
log(1 + popularity_i) > log(1 + popularity_j)
```

Therefore:

```text
Standalone Popularity baseline:
raw popularity ranking

Heuristic two-stage reranker:
log1p(popularity)
→ candidate-level min-max normalization
→ weighted combination with normalized ALS score
```

**Observed result.**
Although log transformation makes the popularity feature numerically more stable, adding static popularity did not improve the current Full MIND whole-catalog ranking results. The `0.99 ALS + 0.01 popularity` configuration performed slightly below direct ALS at `K = 10, 20, 40, and 80`.

This result suggests that the limitation is not caused by the numerical scale of raw popularity. Static training-period popularity itself provides limited value for predicting future news clicks under temporal drift.
