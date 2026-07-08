## 2026-07-05 Experiment: Raw file existence and size check

**Goal.**  
Verify that the MIND-small raw files are placed in the expected project directories and are not failed downloads.

**Files checked.**

```text
../data/raw/MINDsmall_train/behaviors.tsv
../data/raw/MINDsmall_train/news.tsv
../data/raw/MINDsmall_dev/behaviors.tsv
../data/raw/MINDsmall_dev/news.tsv
```

### Method:

I used `Path.exists()` to check whether each file exists and `Path.stat().st_size` to check the file size in bytes.

```python
for path in [
    train_behaviors_path,
    train_news_path,
    dev_behaviors_path,
    dev_news_path,
]:
    print(
        path,
        "exists:",
        path.exists(),
        "size:",
        path.stat().st_size if path.exists() else None,
    )
```

### Results:

```text
../data/raw/MINDsmall_train/behaviors.tsv exists: True size: 92019716
../data/raw/MINDsmall_train/news.tsv exists: True size: 41202121
../data/raw/MINDsmall_dev/behaviors.tsv exists: True size: 42838544
../data/raw/MINDsmall_dev/news.tsv exists: True size: 33519092
```

### Conclusion:

The four required raw files exist and have reasonable sizes. This confirms that the MIND-small raw data has been placed correctly and that these files are not failed downloads or tiny XML error files.

---

## 2026-07-05 Experiment: Load raw MIND-small tables

**Goal.**  
Load `behaviors.tsv` and `news.tsv` for both train and dev splits using pandas.

**Method.**  
The raw files are tab-separated and do not contain header rows, so I manually assigned column names and used:

```python
pd.read_csv(path, sep="\t", names=cols, header=None)
```

The behavior columns are:

```text
impression_id
user_id
time
history
impressions
```

The news columns are:

```text
news_id
category
subcategory
title
abstract
url
title_entities
abstract_entities
```

### Results:

```text
train_behaviors: (156965, 5)
dev_behaviors: (73152, 5)
train_news: (51282, 8)
dev_news: (42416, 8)
```

### Conclusion:

The raw files were loaded successfully. The behavior files have 5 columns, and the news files have 8 columns, matching the expected MIND-small schema.

---

## 2026-07-05 Experiment: Basic dataset statistics

**Goal.**  
Understand the basic scale of the MIND-small train and dev splits before parsing impressions.

**Method.**  
I computed the number of rows and the number of unique users/news items:

```python
summary = {
    "train_behavior_rows": len(train_behaviors),
    "dev_behavior_rows": len(dev_behaviors),
    "train_news_rows": len(train_news),
    "dev_news_rows": len(dev_news),
    "train_unique_users": train_behaviors["user_id"].nunique(),
    "dev_unique_users": dev_behaviors["user_id"].nunique(),
    "train_unique_news": train_news["news_id"].nunique(),
    "dev_unique_news": dev_news["news_id"].nunique(),
}
```

### Results:

```text
train_behavior_rows: 156965
dev_behavior_rows: 73152
train_news_rows: 51282
dev_news_rows: 42416
train_unique_users: 50000
dev_unique_users: 50000
train_unique_news: 51282
dev_unique_news: 42416
```

### Conclusion:

MIND-small contains 50,000 unique users in both train and dev. The number of behavior rows is larger than the number of users because each behavior row corresponds to an impression event, not a unique user. The same user may appear in multiple behavior rows at different times.

---

## 2026-07-05 Experiment: Missing-value check

**Goal.**  
Check missing values in the raw behavior logs and news metadata before parsing impressions or using text features.

**Method.**  
I used:

```python
df.isna().sum()
```

to count missing values in each column.

### Results:

Train behaviors:

```text
impression_id       0
user_id             0
time                0
history          3238
impressions         0
```

Dev behaviors:

```text
impression_id       0
user_id             0
time                0
history          2214
impressions         0
```

Train news:

```text
news_id                 0
category                0
subcategory             0
title                   0
abstract             2666
url                     0
title_entities          3
abstract_entities       4
```

Dev news:

```text
news_id                 0
category                0
subcategory             0
title                   0
abstract             2021
url                     0
title_entities          2
abstract_entities       2
```

### Conclusion:

Missing `history` values are expected because some users may not have prior clicked news before a given impression event. Missing `abstract` values are also present, which matters later if text-based or hybrid models use news abstracts as features. For the current data inspection step, these missing values are documented but do not block impression parsing.

---

## 2026-07-05 Experiment: Inspect raw impression and history examples

**Goal.**  
Inspect the actual string format of `impressions` and `history` to understand how MIND represents user behavior.

**Method.**  
I printed several examples from the `impressions` and `history` columns.

### Example impression strings:

```text
N55689-1 N35729-0
N20678-0 N39317-0 N58114-0 N20495-0 N42977-0 N22407-0 N14592-0 N17059-1 N33677-0 N7821-0 N6890-0
N50014-0 N23877-0 N35389-0 N49712-0 N16844-0 N59685-0 N23814-1 ...
N35729-0 N33632-0 N49685-1 N27581-0
```

### Interpretation:

Each token in the `impressions` column has the format:

```text
news_id-click_label
```

For example:

```text
N55689-1
```

means the news item was exposed and clicked.

```text
N35729-0
```

means the news item was exposed but not clicked.

### Example history interpretation:

The `history` column contains previously clicked news items before the current impression event. Unlike `impressions`, it does not use `-1` or `-0` labels, because the listed items are already past clicked items.

### Conclusion:

The raw MIND behavior logs contain two different types of user behavior information:

```text
history:
  past clicked news before the current impression

impressions:
  candidate news shown in the current impression, with click labels
```

This confirms that the next processing step should parse impressions into event-level user-item-click rows while keeping history conceptually separate.

---

## 2026-07-05 Experiment: Count candidate items per impression

**Goal.**  
Measure how many candidate news items are shown in each impression event.

**Method.**  
The `impressions` column is a whitespace-separated string. I used `split()` to count the number of candidate items in each impression:

```python
train_behaviors["num_impressions"] = train_behaviors["impressions"].fillna("").apply(
    lambda x: len(x.split()) if x else 0
)
```

The name `num_impressions` means the number of candidate items inside one impression string. A more precise future name would be `num_candidate_items`.

### Train results:

```text
count    156965.000000
mean         37.227688
std          38.594148
min           2.000000
25%          10.000000
50%          24.000000
75%          51.000000
max         299.000000
```

### Dev results:

```text
count    73152.000000
mean        37.469898
std         39.541630
min          2.000000
25%         10.000000
50%         23.000000
75%         51.000000
max        295.000000
```

### Conclusion:

Each behavior row contains multiple candidate news items. The median impression contains around 23-24 candidate items, while some impressions contain almost 300 candidate items. This confirms that MIND is naturally suited for ranking and impression-level evaluation.

---

## 2026-07-05 Experiment: Count clicked and non-clicked exposed items

**Goal.**  
Count how many exposed items were clicked and how many exposed items were not clicked in the train and dev impression logs.

**Method.**  
For each token in the `impressions` string:

```text
Nxxxxx-1 means exposed and clicked
Nxxxxx-0 means exposed but not clicked
```

I used a helper function to count clicked and non-clicked items for each impression:

```python
def count_clicks(impression_string):
    if pd.isna(impression_string):
        return 0, 0

    clicked = 0
    non_clicked = 0

    for token in impression_string.split():
        if token.endswith("-1"):
            clicked += 1
        elif token.endswith("-0"):
            non_clicked += 1

    return clicked, non_clicked
```

### Results:

```text
Train clicked items: 236344
Train non-clicked exposed items: 5607100
Dev clicked items: 111383
Dev non-clicked exposed items: 2629615
```

### Conclusion:

The number of exposed but non-clicked items is much larger than the number of clicked items. This confirms that MIND-small is an implicit-feedback dataset with strong class imbalance.

However, exposed non-clicked items should not be treated as explicit dislikes. They mean the user saw the item but did not click it. This is different from completely unobserved user-item pairs that do not appear in the impression logs. Therefore, the parsed interaction table should preserve both clicked and non-clicked impression items, while later modeling choices should distinguish positive clicks, exposed non-clicks, and unobserved pairs.

---

## 2026-07-05 Current status

At this point, the raw MIND-small files have been loaded and inspected. The following parts are complete:

```text
1. Verified raw file paths and file sizes.
2. Loaded train/dev behaviors and news tables.
3. Confirmed the expected schema.
4. Computed basic dataset statistics.
5. Checked missing values.
6. Inspected raw history and impression strings.
7. Counted candidate items per impression.
8. Counted clicked and non-clicked exposed items.
```

The next step is to parse the impression logs into an event-level interaction table with columns such as:

```text
user_id
item_id
click
impression_id
time
source
```


## 2026-07-07 Experiment: Phase 2 Part A Dataset Preparation

**Goal.** Prepare the MIND-small dataset for downstream recommendation modeling by converting raw behavior and news files into cleaned interaction tables.

**Summary.** In Phase 2 Part A, I loaded the MIND-small train/dev behavior files and news metadata, parsed impression logs into user-item-click records, 
added user history clicks as positive interactions, combined and deduplicated train/dev news metadata, and merged news metadata into the interaction tables 
using a left join. After merging, I dropped the duplicate `news_id` column and kept `item_id` as the standard item identifier for downstream recommendation 
models. The train table stayed at `(10951083, 6)` before the metadata merge and became `(10951083, 10)` after merging and dropping `news_id`. The dev table 
stayed at `(5103512, 6)` before the metadata merge and became `(5103512, 10)` after merging and dropping `news_id`. The unchanged row counts confirm that no 
interaction rows were dropped. The final cleaned tables include interaction fields plus news metadata fields such as `category`, `subcategory`, `title`, and `abstract`.

**Generated files.**
- `../data/processed/interactions_history_train.parquet`: train interaction table from parsed impressions and user history.
- `../data/processed/interactions_history_dev.parquet`: dev interaction table from parsed impressions and user history.
- `../data/processed/news.parquet`: combined and deduplicated news metadata from train/dev `news.tsv`.
- `../data/processed/train_with_news.parquet`: cleaned train interaction table enriched with news metadata.
- `../data/processed/dev_with_news.parquet`: cleaned dev interaction table enriched with news metadata.

**Final columns.** `user_id`, `item_id`, `click`, `impression_id`, `time`, `source`, `category`, `subcategory`, `title`, `abstract`.

**Conclusion.** Phase 2 Part A is complete. The cleaned train/dev interaction files are ready for ID mapping and 
sparse user-item matrix construction.

## 2026-07-07 Build Warm-start Dev Matrix for Phase 2 Part B Step 8

**Goal.**
Construct sparse train/dev user-item matrices for classical collaborative filtering models while keeping the validation 
setup consistent with a warm-start recommendation protocol.

**Context.**
In Phase 2 Part A, the processed train/dev interaction tables contain two types of positive signals: historical clicks 
from `source = "history"` and clicked impression items from `source = "impression"`. For training, historical clicks 
and clicked impressions can both be useful positive implicit-feedback signals. For validation, however, the dev matrix 
should represent the future clicked items we want the recommender to recover, so it should use clicked dev impressions 
rather than dev history rows.

**Implementation.**
The train sparse matrix is built from `train_with_news.parquet` using all rows with `click = 1`, including both historical 
clicks and clicked impression rows. The dev sparse matrix is built from `dev_with_news.parquet` using only rows where 
`click = 1` and `source = "impression"`. Before constructing the dev matrix, dev rows are filtered to keep only users 
and items that appear in the train-based ID mappings created in Step 7. This ensures that both train and dev matrices 
use the same row and column index spaces.

**Interpretation.**
The resulting `train_interactions.npz` is the matrix used for model fitting. The resulting `dev_interactions.npz` is 
not used for training; it is a validation ground-truth matrix for later Recall@K, NDCG@K, and MRR evaluation. Dev users 
or dev items that do not appear in train are counted as cold-start cases and excluded from the standard classical 
collaborative filtering evaluation.

**Conclusion.**
For the current classical matrix-based recommender pipeline, Step 8 follows a warm-start evaluation setup: models 
are trained on positive train interactions and evaluated on future dev impression clicks for users and items already 
present in the train matrix.


## 2026-07-07 Experiment: Phase 2 Part B Step 8 Sparse Matrix Construction

**Goal.**
Build train/dev sparse user-item interaction matrices from the cleaned MIND-small interaction tables.

**Method.**
The train matrix was built from `train_with_news.parquet` using all positive interactions where `click = 1`. The dev matrix was built from `dev_with_news.parquet` using clicked dev impressions only, where `click = 1` and `source = "impression"`. Dev pairs were filtered to keep only users and items that appear in the train-based ID mappings. Finally, dev positive `(user_id, item_id)` pairs that already appeared in train positives were removed.

**Results.**

```text
dev clicked impression pairs before filtering: 110745
dev warm-start pairs after user/item filtering: 10314
dev positive pairs already seen in train: 37
dev final positive pairs after removing train-seen pairs: 10277

train matrix shape: (50000, 51282)
dev matrix shape: (50000, 51282)
train nnz: 1148447
dev nnz: 10277
```

**Generated files.**

```text
../data/processed/train_interactions.npz
../data/processed/dev_interactions.npz
```

**Conclusion.**
Step 8 successfully created sparse train/dev interaction matrices with the same user/item index space. The dev matrix is a clean warm-start validation target: it uses clicked dev impressions, excludes cold-start users/items, and removes positive user-item pairs already seen in train.
