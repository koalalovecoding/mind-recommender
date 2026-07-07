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
