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
## 2026-07-09 Experiment: Popularity Baseline on MIND-small

**Goal.**
Implement a non-personalized popularity baseline on the MIND-small training interaction matrix and verify that it can generate top-K recommendations while filtering out items already clicked by the target user.

**Input files.**

```text
../data/processed/train_interactions.npz
../data/processed/idx_user_map.json
../data/processed/idx_item_map.json
../data/processed/news.parquet
```

**Method.**
The training interaction matrix is a binary user-item CSR matrix. Each nonzero entry indicates that a user clicked a news item at least once in the training data.

The popularity score of item (i) is defined as:

$$
\operatorname{popularity}(i)=\sum_u R_{ui}.
$$

Because duplicate user-item interactions were removed when constructing the sparse matrix, the popularity score represents the number of unique training users who clicked each news item.

All items were sorted by popularity score in descending order. To generate recommendations for a user, items already clicked by that user in the training matrix were removed, and the highest-ranked remaining items were returned.

**Implementation.**

```text
1. Load the binary train interaction matrix.
2. Sum the matrix along the user dimension to compute item popularity.
3. Sort all item indices by descending popularity score.
4. Retrieve the target user's previously clicked item indices.
5. Filter previously clicked items from the global popularity ranking.
6. Return the top-K unseen items.
7. Map item indices back to MIND news IDs and titles.
```

### Results

```text
train matrix shape: (50000, 51282)
train matrix nnz: 1148447
items with at least one click: 39865
maximum popularity score: 4747
```

The training matrix contains 50,000 users, 51,282 news items, and 1,148,447 unique positive user-item interactions.

Of the 51,282 news items, 39,865 received at least one training click. The remaining 11,417 items had a popularity score of zero.

The most popular news item was clicked by 4,747 unique training users.

### Global Top-10 Popular News Items

```text
rank  item_idx  news_id  popularity_score
1     17995     N306     4747
2     39853     N55689   4257
3     28487     N42620   3998
4     31231     N45794   3283
5     22430     N35729   3281
6     20568     N33619   3214
7     19019     N31801   3207
8     39409     N55189   3045
9     28935     N43142   2942
10    16737     N29177   2915
```

### Sample User Recommendation

The first user with at least one training interaction was selected as a sample:

```text
sample user_idx: 0
sample user_id: U100
number of training clicks: 11
```

The popularity model successfully generated ten recommendations for this user after filtering items already clicked in the training set.

The recommendation list was identical to the global top-10 popularity ranking because none of the user's 11 previously clicked items appeared among the ten globally most popular items.

### Validation Checks

The following checks passed:

```text
1. The number of matrix rows matched the number of user mappings.
2. The number of matrix columns matched the number of item mappings.
3. Recommended item indices were successfully mapped to MIND news IDs.
4. News IDs were successfully mapped to readable news titles.
5. The sample user's recommendation list contained no item already clicked in train.
6. Popularity scores and rankings were saved successfully.
```

**Generated files.**

```text
../data/processed/popularity_scores.npy
../data/processed/popularity_ranking.npy
```

`popularity_scores.npy` stores the popularity score for every item index.

```text
popularity_scores[item_idx] = number of unique training users who clicked the item
```

`popularity_ranking.npy` stores all item indices ordered from the most popular item to the least popular item.

**Conclusion.**
The MIND-small popularity baseline was implemented successfully. The model calculates global news popularity from training interactions, filters items already clicked by a target user, and returns top-K unseen recommendations.

The model is non-personalized because all users share the same global popularity ranking. User-specific behavior only affects which previously clicked items are removed. This baseline will be used as the minimum comparison model for ALS and the later two-stage recommendation pipeline.

## 2026-07-10 Experiment: Train Implicit ALS on MIND-small

**Goal.**
Train an implicit-feedback Alternating Least Squares model on the MIND-small user-item interaction matrix and verify that the trained model can generate personalized top-K news recommendations.

### Input:

```text
../data/processed/train_interactions.npz
../data/processed/idx_user_map.json
../data/processed/idx_item_map.json
../data/processed/news.parquet
```

The training matrix contains positive implicit-feedback interactions collected from train history clicks and clicked train impressions.

### Training data:

```text
Train matrix shape: (50000, 51282)
Train matrix nonzero entries: 1148447
Train matrix dtype: float32
Stored interaction values: [1.]
```

This represents:

```text
50000 users
51282 news items
1148447 unique positive user-item interactions
```

### Model configuration:

The model was trained using the `AlternatingLeastSquares` implementation from the Python `implicit` library.

```text
factors: 64
regularization: 0.1
alpha: 40.0
iterations: 15
use_cg: True
use_gpu: False
random_state: 42
```

`factors = 64` means that every user and news item is represented by a 64-dimensional latent-factor vector.

`alpha = 40.0` increases the confidence assigned to observed positive interactions.

`regularization = 0.1` penalizes excessively large factor values and helps reduce overfitting.

`iterations = 15` means that ALS performs 15 alternating user-factor and item-factor update rounds. The model stops after 15 rounds because this value was specified manually, not because automatic early stopping was used.

### Method:

The ALS model was fitted directly on the sparse user-item training matrix:

```python
model.fit(train_matrix)
```

During each iteration, ALS alternates between:

```text
1. Fixing item factors and updating user factors.
2. Fixing user factors and updating item factors.
```

The model learns a low-rank approximation of the interaction matrix:

$$
R \approx PQ^\top,
$$

where:

```text
P = user-factor matrix
Q = item-factor matrix
```

The predicted ranking score for user (u) and item (i) is:

$$
s(u,i)=p_u^\top q_i.
$$

### Training results:

```text
ALS training time: approximately 5 seconds
User factors shape: (50000, 64)
Item factors shape: (51282, 64)
User factors dtype: float32
Item factors dtype: float32
```

The factor shapes confirm that the model learned:

```text
one 64-dimensional vector for each of the 50000 users
one 64-dimensional vector for each of the 51282 news items
```

### Sample recommendation results:

The first user with at least one training interaction was selected as the sample user:

```text
user_idx: 0
user_id: U100
unique train clicks: 11
```

Items already clicked by this user in the training data were removed before generating recommendations.

The ALS top-10 recommendations were:

```text
Rank  News ID   ALS score
1     N287      0.914984
2     N4607     0.716596
3     N22816    0.692157
4     N871      0.642523
5     N10470    0.642281
6     N32483    0.636023
7     N4020     0.601750
8     N306      0.593276
9     N60702    0.589527
10    N57737    0.575436
```

The ALS scores are latent-factor ranking scores, not calibrated click probabilities. For example, a score of `0.914984` does not mean that the user has a 91.5% probability of clicking the article. It only means that the article received a higher model score than the other eligible candidate items.

### Validation checks:

The script verified that:

```text
1. The train matrix dimensions match the user and item mappings.
2. The training matrix contains valid positive finite values.
3. The learned user-factor and item-factor shapes are correct.
4. The factor matrices contain no NaN or infinite values.
5. Recommended item indices are within the valid item range.
6. Previously clicked training items are excluded.
7. Returned ALS scores match direct user-item factor dot products.
8. Recommendations are ordered by descending ALS score.
```

### Generated files:

```text
../data/processed/als_model.npz
../data/processed/als_user_factors.npy
../data/processed/als_item_factors.npy
../data/processed/als_config.json
../data/processed/als_sample_top10.csv
```

The complete ALS model can be loaded later without retraining. The separately saved user and item factors will also be used in the FAISS candidate-retrieval pipeline.

### Interpretation:

The experiment confirms that implicit ALS can successfully learn latent user and news-item representations from the MIND-small sparse interaction matrix and produce personalized recommendations.

The sample ALS ranking differs from the global popularity ranking, which shows that the model is using user-specific interaction patterns rather than recommending the same globally popular items to every user.

However, this sample output alone does not prove that ALS performs better than the Popularity baseline. Both models must be evaluated on the same warm-start dev users using Recall@K, NDCG@K, MRR@K, MAP@K, and Hit Rate@K.

### Conclusion:

The ALS training and sample recommendation pipeline ran successfully on MIND-small. The trained model, user factors, and item factors are ready for ranking evaluation and the later ALS → FAISS → reranking pipeline.

## 2026-07-09 Experiment: Whole-Catalog Ranking Evaluation of Popularity and ALS

**Goal.**
Evaluate the Popularity baseline and the implicit ALS model under the same warm-start whole-catalog evaluation protocol, using Recall@K, NDCG@K, MRR@K, MAP@K, and Hit Rate@K.

The experiment compares both models at:

```text
K = 10
K = 20
K = 40
K = 80
```

### Evaluation protocol

The models were trained using:

```text
../data/processed/train_interactions.npz
```

The validation ground truth was loaded from:

```text
../data/processed/dev_interactions.npz
```

The dev matrix contains clicked dev impression items only. Dev users and items not present in the train mappings were removed, and positive user-item pairs already observed in train were excluded from the validation targets.

Therefore, the evaluation tests whether each model can recover future clicked news items for warm-start users and items.

The evaluation uses the entire training item universe as the candidate set:

```text
Number of candidate news items: 51282
```

For each evaluated user:

```text
1. Generate top-K recommendations from all train-known items.
2. Remove items already clicked by the user in train.
3. Compare the remaining recommendations with the user's dev clicked items.
4. Compute the five ranking metrics.
```

This is a whole-catalog evaluation rather than an impression-level evaluation. It is substantially more difficult than ranking only the candidate articles contained in one MIND impression. The Phase 2 plan explicitly uses all items in the training universe as the first evaluation candidate set and leaves impression-level evaluation as a later improvement.

### Metric sanity checks

Before evaluating the models, the metric implementations were tested using manually constructed examples.

The checks confirmed that:

```text
Perfect ranking:
Recall = 1
NDCG = 1
MRR = 1
MAP = 1
Hit Rate = 1

No-hit ranking:
Recall = 0
NDCG = 0
MRR = 0
MAP = 0
Hit Rate = 0
```

Output:

```text
Metric sanity checks passed.
```

### Evaluated users

```text
Evaluated users: 5109
```

Only users with at least one valid dev positive item were included:
- warm-start users/items
- remove train-seen pairs
- keep users with non-empty ground truth 5,109 evaluated users

The dev matrix contains 10,277 final positive user-item pairs, so the evaluated users have approximately:

```text
10277 / 5109 ≈ 2.01
```

relevant dev items per user on average.

### Popularity drift diagnostic

Before running the ranking evaluation, I checked whether the globally most popular train items appeared anywhere among the dev positive items.

Results:

```text
Popularity top-10 items appearing in dev positives: 0
Popularity top-100 items appearing in dev positives: 4
Popularity top-1000 items appearing in dev positives: 42
```

### Interpretation of the diagnostic

None of the ten most popular train items appeared in the entire dev positive-item set.

Only four of the top 100 train-popular items and 42 of the top 1000 appeared in the dev positives.

This indicates strong temporal drift in news popularity:

```text
An article that is popular during the training period may no longer
be relevant or actively clicked during the development period.
```

This effect is especially important for news recommendation because articles have short lifetimes and user attention shifts rapidly toward newer events.

The diagnostic also explains why the Popularity baseline performs poorly at small K.

### Ranking results

```text
     Model  K   Recall     NDCG      MRR      MAP  HitRate
Popularity 10 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 10 0.001783 0.000916 0.000778 0.000546 0.003132
Popularity 20 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 20 0.003258 0.001343 0.000961 0.000651 0.005676
Popularity 40 0.000525 0.000138 0.000051 0.000019 0.001370
       ALS 40 0.005296 0.001831 0.001097 0.000718 0.009787
Popularity 80 0.001242 0.000271 0.000069 0.000032 0.002349
       ALS 80 0.009650 0.002712 0.001243 0.000797 0.018203
```

### Result 1: ALS consistently outperformed Popularity

ALS achieved higher values than Popularity for every metric and every tested K.

At K = 10:

```text
Popularity Recall@10: 0.000000
ALS Recall@10:        0.001783

Popularity HitRate@10: 0.000000
ALS HitRate@10:        0.003132
```

At K = 80:

```text
Popularity Recall@80: 0.001242
ALS Recall@80:        0.009650

Popularity HitRate@80: 0.002349
ALS HitRate@80:        0.018203
```

At K = 80, ALS achieved approximately:

```text
0.009650 / 0.001242 ≈ 7.8
```

times the Recall of Popularity.

Its Hit Rate was approximately:

```text
0.018203 / 0.002349 ≈ 7.7
```

times the Popularity Hit Rate.

This provides quantitative evidence that the personalized latent-factor model performs better than recommending the same globally popular articles to every user under the current protocol.

### Result 2: Popularity produced no hits at K = 10 or K = 20

Popularity returned zero values for all metrics at K = 10 and K = 20.

This is consistent with the popularity-drift diagnostic:

```text
No global top-10 train-popular item appeared in the dev positives.
```

The model recommends almost the same global ranking to every user, with only training-seen items removed. Because the most popular training articles had almost no overlap with future dev clicks, the short recommendation lists contained no relevant validation items.

This does not imply that the Popularity implementation is incorrect. It shows that static training-period popularity is a weak predictor of future news clicks in this temporal setting.

### Result 3: Popularity began producing hits at K = 40

Popularity achieved nonzero results beginning at K = 40:

```text
Recall@40:  0.000525
HitRate@40: 0.001370
```

The diagnostic found four dev-positive items among the global train top 100.

In addition, the Popularity recommendation list is created separately for each user after filtering items already clicked in train. Therefore, a user's top-40 unseen recommendations are not always identical to the first 40 items in the global ranking.

For users who previously clicked some highly popular items, filtering moves lower-ranked popular items into their final top-40 list. This can bring some of the few top-100 items that also appear in dev positives into the recommendation list.

This explains why Popularity can receive nonzero scores at K = 40 even though the global top-10 had no overlap with dev positives.

### Result 4: Larger K increased Recall and Hit Rate

As K increased, both models had more opportunities to include a relevant item.

ALS Hit Rate increased as follows:

```text
HitRate@10: 0.003132
HitRate@20: 0.005676
HitRate@40: 0.009787
HitRate@80: 0.018203
```

Using 5,109 evaluated users, this corresponds approximately to:

```text
K = 10: 5109 × 0.003132 ≈ 16 users with at least one hit
K = 20: 5109 × 0.005676 ≈ 29 users with at least one hit
K = 40: 5109 × 0.009787 ≈ 50 users with at least one hit
K = 80: 5109 × 0.018203 ≈ 93 users with at least one hit
```

The increase is expected because longer recommendation lists cover a larger part of the item catalog.

However, the metric values at different K values should not be compared as though they represent the same task. Recall@80 naturally has an advantage over Recall@10 because the model is allowed to return eight times as many items.

The correct comparison is:

```text
Popularity@10 versus ALS@10
Popularity@20 versus ALS@20
Popularity@40 versus ALS@40
Popularity@80 versus ALS@80
```

### Result 5: MRR and MAP improved more slowly than Recall

For ALS:

```text
MRR@10: 0.000778
MRR@80: 0.001243

MAP@10: 0.000546
MAP@80: 0.000797
```

Recall and Hit Rate increased substantially as K became larger, but MRR and MAP increased much more slowly.

MRR emphasizes the position of the first relevant recommendation. MAP rewards relevant items appearing early and maintaining high precision at the ranks where hits occur.

Therefore, this pattern indicates that many of the additional relevant items found at larger K values were ranked relatively far down the recommendation list.

In other words:

```text
ALS retrieves more relevant items when the list becomes longer,
but many of those items are not yet ranked near the top.
```

This suggests that the retrieval component has some useful personalized signal, while the ranking quality still has significant room for improvement.

### Result 6: Absolute whole-catalog performance remained low

Although ALS consistently outperformed Popularity, its absolute metric values remained low.

The strongest ALS result was:

```text
Recall@80:  0.009650
HitRate@80: 0.018203
```

Even with 80 recommendations, only approximately 1.82% of evaluated users received at least one relevant dev item.

This should not be interpreted as evidence that the ALS implementation failed.

The current task requires the model to identify a small number of future clicked articles from more than 51,000 possible news items. In contrast, a typical MIND impression contains a much smaller candidate set, with a median of approximately 23–24 candidate articles.

Therefore, whole-catalog ranking is a much harder retrieval problem than impression-level ranking.

Other factors that may contribute to the low absolute metrics include:

```text
1. Strong temporal drift and short news lifetimes.
2. Extreme sparsity of user-item interactions.
3. Limited user history for some users.
4. ALS uses collaborative clicks but not article title, category, or recency.
5. The current ALS hyperparameters have not yet been tuned.
6. The squared-error ALS objective does not directly optimize Recall or NDCG.
7. Dev clicks may involve articles whose relevance depends strongly on current events.
```

### Model interpretation

The Popularity model represents a non-personalized baseline:

```text
Every user receives the same global popularity ordering,
except that previously clicked items are removed.
```

The ALS model learns user-specific and item-specific latent factors:

```text
score(u, i) = user_factor[u] · item_factor[i]
```

The consistent improvement of ALS over Popularity shows that collective user-item interaction patterns contain useful personalized information beyond global item popularity.

However, because the absolute metrics remain low, the experiment also shows that collaborative matrix factorization alone is insufficient for strong whole-catalog news recommendation.

Useful future improvements include:

```text
1. ALS hyperparameter tuning.
2. Impression-level candidate evaluation.
3. Article recency features.
4. Category and subcategory features.
5. Text-based item embeddings.
6. A two-stage retrieval and reranking system.
7. Models designed more directly for ranking, such as BPR.
```

### Generated file

```text
../data/processed/ranking_evaluation.csv
```

The file contains one row for each model and K combination:

```text
Popularity, K=10
ALS, K=10
Popularity, K=20
ALS, K=20
Popularity, K=40
ALS, K=40
Popularity, K=80
ALS, K=80
```

### Conclusion

Under the same warm-start whole-catalog evaluation protocol, implicit ALS consistently outperformed the Popularity baseline at K = 10, 20, 40, and 80.

Popularity performed poorly because the articles that were most popular during training had very little overlap with the articles clicked during the dev period, demonstrating strong temporal drift in news popularity.

ALS recovered more relevant items because it used personalized latent user-item interaction patterns rather than a single global ranking. Nevertheless, the low absolute metric values show that selecting future clicked news from the full catalog remains difficult.

The experiment successfully completed the initial ranking-evaluation stage and established a reproducible quantitative baseline for later FAISS retrieval, reranking, content enhancement, and debiasing experiments.


## 2026-07-09 Experiment: Two-Stage ALS–Popularity Reranking Weight Ablation

**Goal.**
Evaluate the MIND-small two-stage recommendation pipeline and determine whether adding global item popularity to the second-stage reranker improves ranking quality over direct ALS recommendation.

The two-stage pipeline is:

```text
ALS user factor
→ FAISS inner-product retrieval
→ filter train-seen items
→ top-100 unseen candidates
→ ALS and popularity reranking
→ final top-K recommendations
```

The second-stage reranking score is:

$$
\text{rerank score}
===================

w_{\mathrm{ALS}}
\cdot
\text{normalized ALS score}
+
w_{\mathrm{pop}}
\cdot
\text{normalized log-popularity},
$$

where:

$$
w_{\mathrm{ALS}}+w_{\mathrm{pop}}=1.
$$

The following configurations were tested:

```text
1.00 ALS + 0.00 Popularity
0.99 ALS + 0.01 Popularity
0.90 ALS + 0.10 Popularity
0.50 ALS + 0.50 Popularity
```

All experiments used:

```text
Dataset: MIND-small
Evaluable warm-start dev users: 5,109
Candidate retrieval: FAISS IndexFlatIP
Candidate set size: 100
Evaluation K values: 10, 20, 40, 80
Metrics: Recall, NDCG, MRR, MAP, and Hit Rate
Retrieval Recall@100: 0.010841
```

---

### Experiment 1: 1.00 ALS + 0.00 Popularity

**Purpose.**
Use ALS as the only second-stage signal. This experiment is a sanity check for the FAISS retrieval, train-seen filtering, reranking, and evaluation logic.

### Results

```text
     Model  K   Recall     NDCG      MRR      MAP  HitRate
Popularity 10 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 10 0.001783 0.000916 0.000778 0.000546 0.003132
  TwoStage 10 0.001783 0.000916 0.000778 0.000546 0.003132

Popularity 20 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 20 0.003258 0.001343 0.000961 0.000651 0.005676
  TwoStage 20 0.003258 0.001343 0.000961 0.000651 0.005676

Popularity 40 0.000525 0.000138 0.000051 0.000019 0.001370
       ALS 40 0.005296 0.001831 0.001097 0.000718 0.009787
  TwoStage 40 0.005296 0.001831 0.001097 0.000718 0.009787

Popularity 80 0.001242 0.000271 0.000069 0.000032 0.002349
       ALS 80 0.009650 0.002712 0.001243 0.000797 0.018203
  TwoStage 80 0.009650 0.002712 0.001243 0.000797 0.018203
```

### Interpretation

The TwoStage results exactly match the direct ALS results for every metric and every value of K.

This is expected because both stages use the same ALS inner-product score:

```text
Stage 1:
FAISS retrieves candidates using the ALS inner product.

Stage 2:
Candidates are reranked using only the ALS score.
```

Selecting the ALS top-100 candidates and then selecting the ALS top-K from those candidates is equivalent to directly selecting the ALS top-K from the complete item catalog.

### Conclusion

This experiment confirms that:

```text
1. FAISS retrieval reproduces the original ALS ranking.
2. Train-seen item filtering is correct.
3. The candidate set preserves the original ALS top-K items.
4. The reranking implementation is correct.
5. The TwoStage and direct ALS evaluations use the same users and ground truth.
```

The `1.00/0.00` configuration is therefore the pipeline-consistency baseline.

---

### Experiment 2: 0.99 ALS + 0.01 Popularity

**Purpose.**
Test whether a very small popularity contribution can improve ranking quality without substantially changing the personalized ALS ranking.

### Results

```text
     Model  K   Recall     NDCG      MRR      MAP  HitRate
Popularity 10 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 10 0.001783 0.000916 0.000778 0.000546 0.003132
  TwoStage 10 0.001783 0.000914 0.000778 0.000542 0.003132

Popularity 20 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 20 0.003258 0.001343 0.000961 0.000651 0.005676
  TwoStage 20 0.003356 0.001367 0.000970 0.000652 0.005872

Popularity 40 0.000525 0.000138 0.000051 0.000019 0.001370
       ALS 40 0.005296 0.001831 0.001097 0.000718 0.009787
  TwoStage 40 0.005492 0.001864 0.001101 0.000719 0.009982

Popularity 80 0.001242 0.000271 0.000069 0.000032 0.002349
       ALS 80 0.009650 0.002712 0.001243 0.000797 0.018203
  TwoStage 80 0.009585 0.002693 0.001239 0.000792 0.018007
```

### Interpretation

At `K=10`, Recall and Hit Rate remain unchanged, while NDCG and MAP decrease slightly. This means that the same number of relevant items is recovered, but some relevant items move to slightly lower positions.

At `K=20` and `K=40`, the TwoStage model produces small improvements in Recall, NDCG, MRR, and Hit Rate.

At `K=80`, the TwoStage model becomes slightly worse than direct ALS.

The improvements are small and are not consistent across all values of K.

### Conclusion

A popularity weight of `0.01` causes only minor ranking changes. It provides a small improvement at `K=20` and `K=40`, but does not consistently outperform ALS.

This configuration should be interpreted as a marginal and unstable result rather than a clear improvement.

---

### Experiment 3: 0.90 ALS + 0.10 Popularity

**Purpose.**
Evaluate whether a more substantial popularity contribution improves the second-stage ranking.

### Results

```text
     Model  K   Recall     NDCG      MRR      MAP  HitRate
Popularity 10 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 10 0.001783 0.000916 0.000778 0.000546 0.003132
  TwoStage 10 0.001669 0.000832 0.000701 0.000490 0.002740

Popularity 20 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 20 0.003258 0.001343 0.000961 0.000651 0.005676
  TwoStage 20 0.003193 0.001288 0.000909 0.000598 0.005676

Popularity 40 0.000525 0.000138 0.000051 0.000019 0.001370
       ALS 40 0.005296 0.001831 0.001097 0.000718 0.009787
  TwoStage 40 0.005264 0.001762 0.001031 0.000664 0.009395

Popularity 80 0.001242 0.000271 0.000069 0.000032 0.002349
       ALS 80 0.009650 0.002712 0.001243 0.000797 0.018203
  TwoStage 80 0.009145 0.002548 0.001164 0.000734 0.017029
```

### Interpretation

The `0.90/0.10` configuration performs worse than direct ALS for almost every metric and every value of K.

At `K=10`, all five ranking metrics decrease. Similar degradation is observed at `K=20`, `K=40`, and `K=80`.

The popularity feature changes the candidate ordering, but these changes generally move relevant items to worse positions or remove them from the evaluated top-K list.

### Conclusion

A popularity weight of `0.10` is too large for this reranking setup. Global item popularity weakens the personalized ALS ranking and reduces recommendation quality.

---

### Experiment 4: 0.50 ALS + 0.50 Popularity

**Purpose.**
Test the effect of assigning equal importance to personalized ALS relevance and global item popularity.

### Results

```text
     Model  K   Recall     NDCG      MRR      MAP  HitRate
Popularity 10 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 10 0.001783 0.000916 0.000778 0.000546 0.003132
  TwoStage 10 0.001294 0.000611 0.000453 0.000371 0.001762

Popularity 20 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 20 0.003258 0.001343 0.000961 0.000651 0.005676
  TwoStage 20 0.002354 0.000918 0.000585 0.000443 0.003719

Popularity 40 0.000525 0.000138 0.000051 0.000019 0.001370
       ALS 40 0.005296 0.001831 0.001097 0.000718 0.009787
  TwoStage 40 0.004260 0.001346 0.000688 0.000505 0.006851

Popularity 80 0.001242 0.000271 0.000069 0.000032 0.002349
       ALS 80 0.009650 0.002712 0.001243 0.000797 0.018203
  TwoStage 80 0.007506 0.002019 0.000813 0.000562 0.013897
```

### Interpretation

Giving popularity the same weight as ALS substantially reduces recommendation quality.

At `K=10`, TwoStage Recall decreases from `0.001783` to `0.001294`, while Hit Rate decreases from `0.003132` to `0.001762`.

The degradation remains substantial at `K=20`, `K=40`, and `K=80`.

This shows that global popularity is not an adequate substitute for personalized ALS relevance.

### Conclusion

The `0.50/0.50` configuration performs substantially worse than direct ALS. Increasing the popularity weight causes the reranker to favor globally popular items at the expense of personalized relevance.

---

## Cross-Experiment Comparison

The experiments show a clear relationship between popularity weight and ranking performance:

```text
Popularity weight = 0.00:
TwoStage exactly reproduces direct ALS.

Popularity weight = 0.01:
Very small and inconsistent changes.
Some improvement at K=20 and K=40, but no consistent gain.

Popularity weight = 0.10:
Recommendation quality generally decreases.

Popularity weight = 0.50:
Recommendation quality decreases substantially.
```

The results indicate that increasing the global popularity contribution progressively weakens personalized ranking quality.

A useful summary at `K=10` is:

```text
ALS weight  Popularity weight  Recall@10  NDCG@10  MRR@10   MAP@10   HitRate@10
1.00        0.00               0.001783   0.000916 0.000778 0.000546 0.003132
0.99        0.01               0.001783   0.000914 0.000778 0.000542 0.003132
0.90        0.10               0.001669   0.000832 0.000701 0.000490 0.002740
0.50        0.50               0.001294   0.000611 0.000453 0.000371 0.001762
```

---

## Overall Conclusion

The two-stage pipeline was implemented correctly:

```text
1. ALS user and item factors were loaded successfully.
2. FAISS inner-product search retrieved ALS-based candidates.
3. Train-seen items were removed.
4. One hundred unseen candidates were retained.
5. Candidates were reranked and evaluated.
6. The ALS-only configuration exactly reproduced direct ALS metrics.
```

However, global item popularity did not provide a reliable second-stage ranking improvement.

The `1.00 ALS + 0.00 Popularity` experiment confirms that the two-stage infrastructure is correct. The later experiments show that adding global popularity produces mixed results at very small weights and clear degradation at larger weights.

The likely reason is that global popularity is neither personalized nor sufficiently time-sensitive for news recommendation. Training-period popularity may favor broadly popular or older articles rather than the specific future articles clicked by individual users.

Therefore, the current results support the following conclusion:

```text
The FAISS two-stage pipeline is correct, but static global popularity
is not an effective reranking feature for this MIND-small setup.
```

More informative second-stage features should be explored later, including:

```text
news recency
category and subcategory match
user recent category preference
title or abstract similarity
user-history text similarity
exposure-based or propensity-based features
```

A learned ranker such as Logistic Regression, LightGBM, or a neural ranking model could then combine these features instead of relying on manually selected weights.



## 2026-07-10 Experiment: Full MIND Raw Data Size Check

**Goal.**  
Verify that the Full MIND train/dev files were downloaded and extracted correctly, and record the raw dataset scale before implementing streaming data processing.

### Files checked

```text
data/raw/MINDlarge_train/behaviors.tsv
data/raw/MINDlarge_train/news.tsv
data/raw/MINDlarge_dev/behaviors.tsv
data/raw/MINDlarge_dev/news.tsv
```

The extracted directories also contain:

```text
entity_embedding.vec
relation_embedding.vec
```

These embedding files are not used in the current collaborative-filtering and ALS pipeline.

### Method

I first inspected the extracted files and their approximate sizes using:

```bash
ls -lh data/raw/MINDlarge_train
ls -lh data/raw/MINDlarge_dev
```

I then counted the number of rows in the behavior and news files using:

```bash
wc -l \
data/raw/MINDlarge_train/behaviors.tsv \
data/raw/MINDlarge_train/news.tsv \
data/raw/MINDlarge_dev/behaviors.tsv \
data/raw/MINDlarge_dev/news.tsv
```

### File sizes

```text
MINDlarge_train/behaviors.tsv           1.3 GB
MINDlarge_train/news.tsv                 81 MB
MINDlarge_train/entity_embedding.vec     38 MB
MINDlarge_train/relation_embedding.vec  1.0 MB

MINDlarge_dev/behaviors.tsv             220 MB
MINDlarge_dev/news.tsv                   56 MB
MINDlarge_dev/entity_embedding.vec       30 MB
MINDlarge_dev/relation_embedding.vec    1.0 MB
```

### Row counts

```text
Train behavior rows: 2,232,748
Train news rows:       101,527

Dev behavior rows:     376,471
Dev news rows:          72,023
```

### Combined raw scale

```text
Total behavior rows: 2,609,219
Total news rows:       173,550
```

Each row in `behaviors.tsv` represents an impression event rather than a unique user. A single behavior row may contain a user history and many candidate news items inside the `impressions` field.

The combined number of news rows is not the final number of unique news items because the train and dev `news.tsv` files may contain overlapping `news_id` values. The final item count will be determined after combining the train/dev news metadata and removing duplicate news IDs.

### Interpretation

The Full MIND behavior files are substantially larger than the MIND-small files. In particular, the train behavior file alone contains more than 2.2 million impression events and occupies approximately 1.3 GB.

Expanding every exposed candidate item, including all `-0` non-clicked impressions, into one in-memory pandas DataFrame would create a much larger interaction table and could exceed local memory.

Therefore, the Full MIND data pipeline should use streaming or chunk-based processing rather than loading and expanding the complete behavior logs in memory at once.

For the initial ALS training matrix, the processing pipeline will extract positive implicit-feedback interactions from:

```text
1. News items in the user history field.
2. Impression items whose click label is 1.
```

Exposed-but-not-clicked items with label `0` will not be inserted as positive entries in the sparse ALS interaction matrix.

### Conclusion

The Full MIND train and dev datasets were downloaded and extracted successfully. The required `behaviors.tsv` and `news.tsv` files exist and have reasonable sizes and row counts.

The raw dataset is ready for streaming interaction processing, train-based user/item mapping construction, sparse matrix generation, and implicit ALS training.


## 2026-07-10 Experiment: Full MIND Streaming Positive-Interaction Processing

**Goal.**  
Process the Full MIND train/dev behavior logs without loading the complete dataset into memory, extract the positive implicit-feedback interactions needed for ALS, and save the results as partitioned Parquet files.

### Input files

```text
data/raw/MINDlarge_train/behaviors.tsv
data/raw/MINDlarge_dev/behaviors.tsv
```

### Processing strategy

The Full MIND behavior files are too large to expand into one complete in-memory pandas DataFrame.

The script therefore reads each `behaviors.tsv` file line by line and keeps only a limited number of extracted interactions in memory.

The buffer size was configured as:

```text
500,000 interaction rows
```

When the buffer reached approximately 500,000 rows, the script:

```text
1. Converted the buffered interactions into a pandas DataFrame.
2. Saved the DataFrame as one compressed Parquet part file.
3. Cleared the in-memory buffers.
4. Continued processing the remaining behavior rows.
```

This streaming design prevents memory usage from growing with the total dataset size.

### Positive-interaction definitions

For the training split, the script extracted:

```text
1. Every item appearing in the user's history field.
2. Every item in the impressions field with click label = 1.
```

For the development split, the script extracted:

```text
1. Only items in the impressions field with click label = 1.
```

Dev history was intentionally excluded because it represents previous user behavior rather than the future validation targets that the recommender should recover.

Impression items with click label `0` were not saved in this step because the current goal is to construct a positive implicit-feedback sparse matrix for ALS.

### Command

The script was executed from the project root using:

```bash
python -u notebooks/11_parse_mind_large_streaming.py
```

The `-u` option runs Python with unbuffered standard output, so progress messages and errors appear in the terminal immediately.

### Train results

```text
Raw behavior rows:                 2,232,748
History interaction occurrences: 73,629,868
Clicked impression occurrences:   3,383,656
Total positive rows written:     77,013,524
Parquet part files:                     155
Malformed behavior rows:                   0
Malformed impression tokens:               0
Processing time:                        30.23 seconds
```

### Dev results

```text
Raw behavior rows:                   376,471
History interaction occurrences:          0
Clicked impression occurrences:      574,845
Total positive rows written:         574,845
Parquet part files:                        2
Malformed behavior rows:                  0
Malformed impression tokens:              0
Processing time:                        1.99 seconds
```

### Why the train output contains 155 Parquet files

The train split produced:

```text
77,013,524 positive interaction rows
```

The configured buffer size was:

```text
500,000 rows per Parquet part
```

Therefore:

```text
77,013,524 / 500,000 ≈ 154.03
```

The output consists of:

```text
154 files containing approximately 500,000 rows each
1 final file containing the remaining rows
155 Parquet files in total
```

The files are named sequentially:

```text
part-00000.parquet
part-00001.parquet
part-00002.parquet
...
part-00154.parquet
```

These 155 files are not separate datasets. Together, they form one logical Full MIND training positive-interaction dataset.

The dev split produced two files because:

```text
574,845 / 500,000 ≈ 1.15
```

Therefore, the dev output contains:

```text
1 file with approximately 500,000 rows
1 file with the remaining 74,845 rows
```

### Generated files

```text
data/processed/mindlarge/train_positive_interactions/
    part-00000.parquet
    part-00001.parquet
    ...
    part-00154.parquet

data/processed/mindlarge/dev_clicked_impressions/
    part-00000.parquet
    part-00001.parquet

data/processed/mindlarge/11_streaming_parse_summary.json
```

Each Parquet interaction row contains:

```text
user_id
item_id
click
source
```

All saved rows have:

```text
click = 1
```

The `source` field is:

```text
history
```

or:

```text
impression
```

### Important interpretation

The value:

```text
77,013,524
```

represents positive interaction occurrences, not unique user-item pairs.

The same `(user_id, item_id)` pair may appear multiple times, especially because a previously clicked news item can appear repeatedly in a user's history across multiple behavior events.

For example:

```text
Behavior 1 history: N1 N2
Behavior 2 history: N1 N2 N3
Behavior 3 history: N1 N2 N3 N4
```

The pair involving `N1` may be written three times during streaming, but the final binary interaction matrix should still contain:

```text
R[user, N1] = 1
```

rather than a value of 3.

Global deduplication was intentionally not performed during streaming because maintaining all unique user-item pairs in one in-memory Python object would undermine the memory-safe processing design.

Duplicate coordinates will be consolidated later when constructing the binary sparse user-item matrix.

### Comparison with MIND-small processing

For MIND-small, the expanded interaction tables could be loaded and processed as complete pandas DataFrames.

The MIND-small pipeline could directly use operations such as:

```python
drop_duplicates(subset=["user_id", "item_id"])
```

over the complete interaction table.

For Full MIND, the train behavior file contains more than 2.2 million behavior events, and the positive-interaction extraction alone produced more than 77 million rows.

Therefore, the Full MIND pipeline uses:

```text
line-by-line input reading
→ bounded in-memory buffers
→ multiple Parquet output parts
→ delayed global deduplication
→ later sparse matrix construction
```

The semantic definitions remain consistent with the MIND-small pipeline:

```text
Train:
history clicks + clicked train impressions

Dev ground truth:
clicked dev impressions only
```

The difference is primarily an engineering and memory-management change required to scale the pipeline.

### Validation checks

The following checks passed:

```text
1. All 2,232,748 train behavior rows were processed.
2. All 376,471 dev behavior rows were processed.
3. The number of rows written matched the number of extracted interactions.
4. No malformed behavior rows were detected.
5. No malformed impression tokens were detected.
6. Train history and clicked impressions were both extracted.
7. Dev history was excluded.
8. Output files were saved as partitioned Parquet datasets.
9. A JSON processing summary was saved successfully.
```

### Conclusion

The Full MIND train and dev behavior logs were processed successfully using a memory-safe streaming pipeline.

The script extracted:

```text
77,013,524 train positive interaction occurrences
574,845 dev clicked-impression occurrences
```

and saved them as partitioned Parquet files without loading the complete expanded dataset into memory.

The outputs are ready for train-based user/item mapping construction, global user-item deduplication, binary sparse matrix generation, and implicit ALS training.





## 2026-07-10 Experiment: Build Full MIND User and Item Mappings

**Goal.**  
Create train-based integer mappings for all Full MIND users and news items.

### Method

The mappings were built directly from the raw training files:

```text
MINDlarge_train/behaviors.tsv → unique user IDs
MINDlarge_train/news.tsv      → unique news IDs
```

The IDs were sorted before assigning indices so that the mappings are deterministic and reproducible.

### Results

```text
Train behavior rows: 2,232,748
Unique train users:   711,222

Train news rows:      101,527
Unique train items:   101,527

Expected matrix shape:
(711,222, 101,527)

Processing time:
approximately 3.2 seconds
```

No malformed behavior or news rows were found.

### Generated files

```text
data/processed/mindlarge/user_idx_map.json
data/processed/mindlarge/item_idx_map.json
data/processed/mindlarge/idx_user_map.json
data/processed/mindlarge/idx_item_map.json
```

### Conclusion

The Full MIND train user and item spaces were 
mapped successfully. The same global mappings 
will be used across all Parquet interaction 
partitions when constructing the sparse user-item matrix.

## 2026-07-10 Experiment: Build Full MIND Sparse Interaction Matrices

**Goal.**  
Construct binary train/dev CSR user-item matrices for Full MIND using the train-based mappings.

### Method

The 155 train Parquet parts were processed in batches of 10. Each batch was mapped to integer user/item indices, converted into a binary sparse matrix, and merged using element-wise maximum so repeated user-item pairs remained equal to 1.

The dev matrix used clicked dev impressions only. Cold-start users/items and train-seen user-item pairs were removed.

### Results

```text
Matrix shape: (711,222, 101,527)

Train:
Parquet parts:          155
Raw positive rows:      77,013,524
Unique positive pairs:  16,532,504
Density:                0.0002289559

Dev:
Parquet parts:          2
Raw clicked rows:       574,845
Warm-start rows:        463,600
Cold-start users:       39,212
Cold-start items:       1,359
Train-seen pairs removed: 1,702
Final positive pairs:   459,068
Density:                0.0000063576

Processing time:        56.62 seconds
```

### Generated files

```text
data/processed/mindlarge/train_interactions.npz
data/processed/mindlarge/dev_interactions.npz
```

### Conclusion

The Full MIND sparse matrices were constructed successfully.
 Duplicate positive occurrences were consolidated into binary 
 user-item pairs, and the dev matrix follows the same warm-start evaluation protocol used for MIND-small.


## 2026-07-10 Experiment: Train Implicit ALS on Full MIND

**Goal.**  
Train an implicit-feedback ALS model on the Full MIND binary user-item interaction matrix and save the learned user/item latent factors.

### Training data

```text
Train matrix shape: (711,222, 101,527)
Unique positive pairs: 16,532,504
Matrix dtype: float32
```

### Model configuration

```text
factors:        64
regularization: 0.1
alpha:          40.0
iterations:     15
random_state:   42
use_gpu:        False
```

### Results

```text
Training time:       88.91 seconds
Average time/round:   approximately 5.91 seconds

User factors shape:  (711,222, 64)
Item factors shape:  (101,527, 64)
Factor dtype:        float32
```

The model learned one 64-dimensional latent vector for every train user and every train news item.

### Generated files

```text
data/processed/mindlarge/als_model.npz
data/processed/mindlarge/als_user_factors.npy
data/processed/mindlarge/als_item_factors.npy
```

### Conclusion

Implicit ALS was trained successfully on Full MIND. The saved user and item factors can now be used for personalized top-K recommendation, FAISS candidate retrieval, and ranking evaluation.


## 2026-07-10 Experiment: Full MIND Popularity and ALS Sample Recommendations

**Goal.**
Verify that the Full MIND interaction matrix and saved ALS factors can generate valid Popularity and personalized ALS recommendations.

**Input files.**

```text
data/processed/mindlarge/train_interactions.npz
data/processed/mindlarge/als_user_factors.npy
data/processed/mindlarge/als_item_factors.npy
data/processed/mindlarge/idx_user_map.json
data/processed/mindlarge/idx_item_map.json
data/raw/MINDlarge_train/news.tsv
```

**Method.**

```text
1. Loaded the Full MIND binary train interaction matrix.
2. Calculated item popularity as the number of unique train users who clicked each item.
3. Selected the first user with a nonempty train history.
4. Generated the user's top-10 unseen Popularity recommendations.
5. Computed ALS scores using the user-item factor dot product.
6. Removed items already clicked by the user in train.
7. Generated the user's top-10 unseen ALS recommendations.
8. Mapped item indices back to news IDs and titles.
9. Saved the popularity features and sample recommendation results.
```

The ALS score was computed as:

```text
ALS score(u, i) = user_factor[u] · item_factor[i]
```

The score is a ranking score, not a calibrated click probability.

### Results

```text
Train matrix shape: (711222, 101527)
Sample user index: 0
Sample user ID: U0
Sample user train clicks: 11
Popularity recommendations generated: 10
ALS recommendations generated: 10
```

The Popularity model recommended globally popular unseen news items, while ALS produced a personalized ranking based on the latent factor representation of user `U0`.

For the sample user U0, only one item appeared in both top-10 lists, showing that the personalized ALS ranking differed substantially from the Popularity ranking for this example.

### Validation checks

```text
1. User-factor dimensions matched the train matrix users.
2. Item-factor dimensions matched the train matrix items.
3. Train-seen items were excluded from both recommendation lists.
4. ALS scores matched direct user-item factor dot products.
5. Item indices were successfully mapped to news IDs and titles.
6. Both output lists contained ten recommendations.
```

**Generated files.**

```text
data/processed/mindlarge/popularity_scores.npy
data/processed/mindlarge/popularity_ranking.npy
data/processed/mindlarge/popularity_sample_top10.csv
data/processed/mindlarge/als_sample_top10.csv
```

**Conclusion.**
Step 15 was completed successfully. The Full MIND pipeline can now calculate item popularity and generate valid unseen top-10 recommendations using both the non-personalized Popularity baseline and personalized ALS latent-factor scores. The saved outputs are ready for Full MIND ranking evaluation and the later FAISS two-stage pipeline.



## 2026-07-10 Experiment: Full MIND Popularity and ALS Ranking Evaluation

**Goal.**
Evaluate the Full MIND Popularity baseline and implicit ALS model under the same warm-start whole-catalog ranking protocol.

**Evaluation setup.**

```text
Train matrix shape: (711222, 101527)
Dev positive pairs: 459068
Evaluated users: 205536
Candidate universe: 101527 train-known news items
K values: 10, 20, 40, 80
Metrics: Recall, NDCG, MRR, MAP, Hit Rate
```

Only users with at least one valid dev positive item were included. Items already clicked by each user in train were removed from the recommendation results.

### Results

```text
     Model  K   Recall     NDCG      MRR      MAP  HitRate
Popularity 10 0.000000 0.000000 0.000000 0.000000 0.000000
       ALS 10 0.001172 0.000670 0.000800 0.000359 0.002632

Popularity 20 0.000007 0.000003 0.000002 0.000000 0.000034
       ALS 20 0.002250 0.001004 0.000972 0.000430 0.005211

Popularity 40 0.000779 0.000200 0.000067 0.000028 0.001786
       ALS 40 0.004535 0.001578 0.001148 0.000510 0.010319

Popularity 80 0.001092 0.000262 0.000077 0.000034 0.002394
       ALS 80 0.008415 0.002398 0.001299 0.000578 0.019053
```

### Interpretation

ALS outperformed the static Popularity baseline for every metric and every tested value of K.

At `K = 80`:

```text
ALS Recall@80:        0.008415
Popularity Recall@80: 0.001092

ALS HitRate@80:        0.019053
Popularity HitRate@80: 0.002394
```

ALS achieved approximately 7.7 times the Recall and 8.0 times the Hit Rate of Popularity at `K = 80`.

Popularity produced almost no hits at small K, indicating that globally popular train-period news items had very limited overlap with future dev clicks. ALS performed better because it used personalized latent user-item interaction patterns.

Absolute metric values remained low because this was a whole-catalog retrieval task over more than 100,000 news items, with strong sparsity and temporal drift.

**Generated file.**

```text
data/processed/mindlarge/ranking_evaluation.csv
```

**Conclusion.**
The Full MIND ranking evaluation completed successfully. Personalized ALS consistently outperformed the static Popularity baseline, establishing the direct ALS results that will be used as the baseline for the Full MIND FAISS two-stage pipeline.



## 2026-07-10 Experiment: Full MIND Two-Stage FAISS Retrieval and Heuristic Reranking

**Goal.**
Build and evaluate a Full MIND two-stage recommendation pipeline using ALS latent factors for FAISS candidate retrieval and a heuristic ALS-plus-popularity reranker.

The pipeline is:

```text
ALS user factor
→ FAISS inner-product retrieval
→ remove train-seen items
→ retain top-100 unseen candidates
→ heuristic reranking
→ evaluate final top-K recommendations
```

The experiment also directly compares:

```text
Popularity
ALS
TwoStageHeuristic
```

at:

```text
K = 10
K = 20
K = 40
K = 80
```

---

### Input files

```text
data/processed/mindlarge/train_interactions.npz
data/processed/mindlarge/dev_interactions.npz
data/processed/mindlarge/als_user_factors.npy
data/processed/mindlarge/als_item_factors.npy
data/processed/mindlarge/popularity_scores.npy
data/processed/mindlarge/ranking_evaluation.csv
data/processed/mindlarge/idx_user_map.json
data/processed/mindlarge/idx_item_map.json
data/raw/MINDlarge_train/news.tsv
```

---

### Evaluation setup

The experiment used the same warm-start whole-catalog evaluation protocol as Step 16.

```text
Train-known users:        711,222
Train-known items:        101,527
Valid dev positive pairs: 459,068
Evaluated users:          205,536
```

Only users with at least one valid dev positive interaction were evaluated.

For each user:

```text
1. Use the ALS user factor as the FAISS query vector.
2. Search the Full MIND item-factor index.
3. Retrieve extra items so that 100 unseen candidates remain after filtering.
4. Remove items already clicked by the user in train.
5. Retain exactly 100 unique unseen candidates.
6. Recompute exact ALS scores for the candidates.
7. Add a log-transformed popularity feature.
8. Rerank the 100 candidates.
9. Evaluate the reranked list at K = 10, 20, 40, and 80.
```

---

### FAISS index

The item-factor matrix was indexed using:

```python
faiss.IndexFlatIP
```

The index performs exact maximum-inner-product search.

Because ALS scores are calculated as:

```text
score(u, i) = user_factor[u] · item_factor[i]
```

the FAISS inner-product score should match the original ALS dot-product score.

### FAISS validation result

```text
FAISS index build time: 0.0144 seconds
Maximum FAISS score error: 5.960464477539063e-08
```

The maximum numerical difference was far below the validation tolerance of `1e-5`.

This confirms that:

```text
FAISS inner-product retrieval
≈
direct ALS user-item dot-product scoring
```

---

### Heuristic reranker

The reranker used:

```text
ALS weight:        0.99
Popularity weight: 0.01
```

For each user, the candidate features were calculated as:

```text
normalized ALS score
normalized log-popularity score
```

The popularity feature was:

```python
np.log1p(popularity[candidate_items])
```

The final heuristic score was:

```text
rerank_score
=
0.99 × normalized_ALS
+
0.01 × normalized_log_popularity
```

The logarithmic transformation compresses extreme popularity values before min-max normalization and prevents a small number of highly popular articles from dominating the combined score.

---

### ALS-only pipeline sanity check

Before evaluating the heuristic configuration, the reranker was tested using:

```text
ALS weight:        1.0
Popularity weight: 0.0
```

The two-stage pipeline exactly reproduced the direct ALS metrics at all evaluated values of K.

```text
ALS-only pipeline sanity check passed.
```

This validates that:

```text
FAISS retrieval
→ train-seen filtering
→ top-100 candidate construction
→ reranking
```

does not change the original ALS ranking when the popularity feature is disabled.

Therefore, any metric difference in the `0.99 + 0.01` experiment is caused by the popularity reranking component rather than by FAISS retrieval.

---

### Candidate retrieval result

```text
Candidate Recall@100: 0.010355
```

Candidate Recall@100 measures the fraction of valid dev relevant items contained anywhere in the 100-item FAISS candidate set before final reranking.

The value is identical for `K = 10, 20, 40, and 80` because all final evaluations use the same top-100 candidate set.

---

### Two-stage ranking results

```text
            Model  K  EvaluatedUsers   Recall     NDCG      MRR      MAP  HitRate  CandidateRecall@100  ALSWeight  PopularityWeight
TwoStageHeuristic 10          205536 0.001158 0.000662 0.000791 0.000354 0.002613             0.010355       0.99              0.01
TwoStageHeuristic 20          205536 0.002241 0.000998 0.000964 0.000426 0.005196             0.010355       0.99              0.01
TwoStageHeuristic 40          205536 0.004513 0.001568 0.001138 0.000505 0.010246             0.010355       0.99              0.01
TwoStageHeuristic 80          205536 0.008390 0.002387 0.001289 0.000574 0.018999             0.010355       0.99              0.01
```

As K increased, Recall and Hit Rate also increased because longer recommendation lists had more opportunities to include a relevant item.

```text
Recall@10: 0.001158
Recall@20: 0.002241
Recall@40: 0.004513
Recall@80: 0.008390
```

```text
HitRate@10: 0.002613
HitRate@20: 0.005196
HitRate@40: 0.010246
HitRate@80: 0.018999
```

Recall@80 remained below Candidate Recall@100:

```text
Recall@80:             0.008390
Candidate Recall@100:  0.010355
```

This indicates that some relevant items were successfully retrieved into the top-100 candidate set but remained between ranks 81 and 100 after reranking.

---

### Popularity versus ALS versus TwoStage

```text
            Model  K  EvaluatedUsers   Recall     NDCG      MRR      MAP  HitRate  CandidateRecall@100  ALSWeight  PopularityWeight
       Popularity 10          205536 0.000000 0.000000 0.000000 0.000000 0.000000                  NaN        NaN               NaN
              ALS 10          205536 0.001172 0.000670 0.000800 0.000359 0.002632                  NaN        NaN               NaN
TwoStageHeuristic 10          205536 0.001158 0.000662 0.000791 0.000354 0.002613             0.010355       0.99              0.01

       Popularity 20          205536 0.000007 0.000003 0.000002 0.000000 0.000034                  NaN        NaN               NaN
              ALS 20          205536 0.002250 0.001004 0.000972 0.000430 0.005211                  NaN        NaN               NaN
TwoStageHeuristic 20          205536 0.002241 0.000998 0.000964 0.000426 0.005196             0.010355       0.99              0.01

       Popularity 40          205536 0.000779 0.000200 0.000067 0.000028 0.001786                  NaN        NaN               NaN
              ALS 40          205536 0.004535 0.001578 0.001148 0.000510 0.010319                  NaN        NaN               NaN
TwoStageHeuristic 40          205536 0.004513 0.001568 0.001138 0.000505 0.010246             0.010355       0.99              0.01

       Popularity 80          205536 0.001092 0.000262 0.000077 0.000034 0.002394                  NaN        NaN               NaN
              ALS 80          205536 0.008415 0.002398 0.001299 0.000578 0.019053                  NaN        NaN               NaN
TwoStageHeuristic 80          205536 0.008390 0.002387 0.001289 0.000574 0.018999             0.010355       0.99              0.01
```

---

### Result 1: ALS outperformed Popularity

Direct ALS achieved substantially higher values than the static Popularity baseline at every evaluated K.

At `K = 80`:

```text
Popularity Recall@80: 0.001092
ALS Recall@80:        0.008415
```

ALS achieved approximately:

```text
0.008415 / 0.001092 ≈ 7.7
```

times the Recall of Popularity.

For Hit Rate:

```text
Popularity HitRate@80: 0.002394
ALS HitRate@80:        0.019053
```

ALS achieved approximately eight times the Hit Rate of Popularity.

This confirms that personalized latent-factor scores provide substantially more useful ranking signal than static global train-period popularity.

---

### Result 2: The heuristic reranker did not improve direct ALS

The `0.99 ALS + 0.01 popularity` configuration performed slightly below direct ALS at every value of K.

At `K = 10`:

```text
ALS Recall@10:        0.001172
TwoStage Recall@10:   0.001158

ALS NDCG@10:          0.000670
TwoStage NDCG@10:     0.000662
```

At `K = 80`:

```text
ALS Recall@80:        0.008415
TwoStage Recall@80:   0.008390

ALS NDCG@80:          0.002398
TwoStage NDCG@80:     0.002387
```

The differences were small, but consistently negative across Recall, NDCG, MRR, MAP, and Hit Rate.

The correct conclusion is:

```text
Adding a 1% static popularity contribution slightly degraded
ranking quality compared with direct ALS under the current
Full MIND whole-catalog evaluation protocol.
```

---

### Result 3: The performance difference came from reranking, not retrieval

The ALS-only sanity check reproduced direct ALS exactly.

Therefore, FAISS candidate retrieval did not cause the observed metric reduction.

The reduction occurred because the popularity feature changed the ordering of some candidates near the evaluation boundaries:

```text
rank 10 versus rank 11
rank 20 versus rank 21
rank 40 versus rank 41
rank 80 versus rank 81
```

Even a small popularity weight can change these positions because both ALS and popularity features are independently min-max normalized within each user's 100-item candidate set.

---

### Result 4: Static popularity was not useful for future news ranking

The standalone Popularity baseline was weak, especially at small K:

```text
Popularity Recall@10: 0.000000
Popularity Recall@20: 0.000007
```

News popularity changes rapidly over time. An article that received many clicks during the training period may no longer be current or relevant during the dev period.

Therefore, static cumulative popularity may promote historically popular but temporally stale articles.

The experiment suggests that the limitation is not primarily caused by numerical scaling because the popularity feature already used:

```text
log transformation
→ candidate-level normalization
```

Instead, static train-period popularity itself contains limited predictive information for future news clicks.

Possible future improvements include:

```text
time-decayed popularity
recent-window popularity
article recency
category-aware popularity
user-specific category preference
learned reranking
```

---

### Sample recommendation output

The pipeline generated and saved a readable top-10 recommendation list for sample user:

```text
user_idx: 0
user_id:  U0
```

For this user, the final top-10 retained the same retrieval ranks as the original ALS order:

```text
retrieval ranks 1 through 10
```

This means the `1%` popularity feature did not change this sample user's first ten recommendations, although it changed rankings for some other users in the full evaluation.

---

### Latency results

```text
FAISS index build time:             0.0144 seconds
Candidate retrieval time:          42.4406 seconds
Candidate retrieval latency:        0.2065 ms/user
Filtering and reranking time:       16.3308 seconds
Filtering and reranking latency:     0.0795 ms/user
End-to-end evaluation time:         80.8330 seconds
End-to-end latency:                  0.3933 ms/user
```

The heuristic reranker added only a small amount of computation:

```text
0.0795 ms per user
```

The FAISS retrieval stage required approximately:

```text
0.2065 ms per user
```

The reported end-to-end latency also includes Python control flow and ranking-metric evaluation, so it should not be interpreted as pure online serving latency.

Nevertheless, the experiment demonstrates that the Full MIND two-stage pipeline can process recommendations at sub-millisecond average time per evaluated user in the current offline implementation.

---

### Generated files

```text
data/processed/mindlarge/two_stage_candidates_sample.csv
data/processed/mindlarge/two_stage_top10.csv
data/processed/mindlarge/two_stage_evaluation.csv
data/processed/mindlarge/two_stage_model_comparison.csv
data/processed/mindlarge/two_stage_latency.json
```

`two_stage_candidates_sample.csv` contains the 100 FAISS candidates for the sample user.

`two_stage_top10.csv` contains the final readable top-10 recommendation list.

`two_stage_evaluation.csv` contains the TwoStage metrics at:

```text
K = 10, 20, 40, 80
```

`two_stage_model_comparison.csv` contains the direct comparison among:

```text
Popularity
ALS
TwoStageHeuristic
```

`two_stage_latency.json` stores index construction, retrieval, reranking, and end-to-end latency measurements.

---

### Conclusion

The Full MIND two-stage recommendation pipeline was implemented successfully.

The experiment verified that:

```text
1. FAISS inner-product scores accurately reproduce ALS dot products.
2. Every evaluated user receives 100 unique unseen candidates.
3. The ALS-only two-stage configuration reproduces direct ALS.
4. Candidate Recall@100 is 0.010355.
5. The pipeline evaluates final rankings at K = 10, 20, 40, and 80.
6. ALS outperforms the static Popularity baseline at every K.
7. Adding 1% log-popularity slightly reduces ranking quality.
8. The pipeline runs with sub-millisecond average offline latency per user.
```

The best current heuristic configuration remains:

```text
ALS weight:        1.0
Popularity weight: 0.0
```

However, the `0.99 + 0.01` experiment is still valuable because it demonstrates that adding a simple static popularity feature does not automatically improve news recommendation.

The next step is to replace manually selected heuristic weights with a learned PyTorch reranker using candidate-level features and impression-level click labels.



## 2026-07-10 Experiment: Full MIND Learned Reranker with Separate FAISS and PyTorch Processes

**Goal.**
Complete the Full MIND learned-reranking pipeline by training a PyTorch MLP on impression-level positive and negative samples, applying it to FAISS top-100 candidates, and comparing the final ranking quality with Popularity, direct ALS, and the heuristic two-stage model.

A secondary goal was to avoid the macOS OpenMP conflict caused by importing PyTorch and FAISS in the same Python process.

---

### Pipeline architecture

The final Step 18 pipeline was divided into three logical stages:

```text
Training stage:
Full MIND train impressions
→ same-impression positive/negative sampling
→ five candidate-level features
→ original-order 90/10 split
→ feature standardization
→ PyTorch MLP training
→ best checkpoint

Step 18a:
ALS user/item factors
→ FAISS IndexFlatIP
→ top-100 unseen candidates
→ save candidate arrays

Step 18b:
load top-100 candidate arrays
→ construct reranking features
→ load PyTorch checkpoint
→ learned candidate scoring
→ final top-K ranking
→ dev evaluation
→ four-model comparison
```

FAISS and PyTorch were intentionally executed in separate processes.

---

### Ranker training data

The official Full MIND train behavior rows were split in original order:

```text
Total behavior rows:      2,232,748
Ranker train rows:        2,009,473
Ranker validation rows:     223,275
```

For each impression:

```text
1. All mapped clicked candidates were retained as positive samples.
2. Up to four exposed-but-not-clicked candidates were sampled per positive.
3. Negative candidates were sampled from the same impression.
4. The current impression history was used for user-profile features.
```

Final candidate-level sample counts:

```text
Ranker train samples:      14,536,251
Ranker validation samples:  1,615,970
```

The samples were stored in memory-safe NumPy parts:

```text
Train parts:       30
Validation parts:   4
Total parts:       34
Samples per full part: 500,000
```

---

### Ranking features

The learned reranker used five features:

```text
1. ALS user-item dot-product score
2. log1p item popularity
3. user history length
4. category affinity
5. subcategory affinity
```

Category affinity was defined as:

```text
number of historical clicks in the candidate category
-----------------------------------------------------
mapped user-history length
```

Subcategory affinity was calculated analogously.

Feature standardization statistics were computed using ranker-training samples only.

### Feature statistics

```text
Feature                  Mean        Standard deviation
ALS score                0.335474    0.308629
log1p popularity         7.630261    1.713575
history length          40.482110   49.948868
category affinity        0.167237    0.198416
subcategory affinity     0.044205    0.088190
```

---

### PyTorch model

The pointwise reranker used a small MLP:

```text
Input features: 5
Hidden layer 1: 32 units + ReLU + dropout
Hidden layer 2: 16 units + ReLU + dropout
Output: 1 logit
```

Training configuration:

```text
Loss: BCEWithLogitsLoss
Optimizer: AdamW
Learning rate: 0.001
Weight decay: 0.00001
Dropout: 0.10
Batch size: 8192
Maximum epochs: 8
Negative ratio: 1:4
Device: Apple MPS
```

### Training results

```text
Epoch  Train loss  Validation loss
1      0.366178    0.342743
2      0.345559    0.340929
3      0.344005    0.340335
4      0.343337    0.339846
5      0.342891    0.339599
6      0.342611    0.339413
7      0.342288    0.339287
8      0.342100    0.338916
```

Best checkpoint:

```text
Best epoch: 8
Best validation loss: 0.3389157276458985
Training time: 59.90 seconds
```

Both training loss and validation loss decreased throughout training, and the epoch-8 checkpoint was retained.

---

## Step 18a: FAISS top-100 candidate retrieval

**Method.**
Step 18a loaded the saved ALS factors, built an exact inner-product FAISS index, retrieved candidates for every evaluable dev user, filtered train-seen items, and saved exactly 100 unseen candidates per user.

Step 18a imported FAISS but did not import PyTorch.

### Retrieval results

```text
Evaluated users:             205,536
Indexed items:               101,527
Candidates per user:             100
Candidate Recall@100:       0.010355
FAISS index build time:     0.020937 seconds
Candidate retrieval time:  83.889672 seconds
Retrieval latency:          0.408151 ms/user
```

Candidate Recall@100 measures the fraction of valid dev positive items contained anywhere in the FAISS top-100 candidate set.

The candidate arrays were saved as `int32`, requiring approximately:

```text
205,536 × 100 × 4 bytes ≈ 82 MB
```

### Generated Step 18a files

```text
data/processed/mindlarge/learned_reranker_eval_users.npy
data/processed/mindlarge/learned_reranker_faiss_candidates.npy
data/processed/mindlarge/learned_reranker_sample_user.npy
data/processed/mindlarge/learned_reranker_sample_candidates.npy
data/processed/mindlarge/learned_reranker_sample_faiss_scores.npy
data/processed/mindlarge/learned_reranker_retrieval_summary.json
```

---

## Step 18b: PyTorch learned reranking and evaluation

**Method.**
Step 18b loaded the candidates created by Step 18a, constructed the five inference features, standardized them using the statistics stored in the checkpoint, calculated MLP logits, reranked each user's 100 candidates, and evaluated the final ranking against official dev positives.

Step 18b imported PyTorch but did not import FAISS.

Dev labels were used only after:

```text
candidate retrieval
→ feature construction
→ MLP scoring
→ final candidate sorting
```

They were not used to train the ranker or construct its input features.

### Learned-reranker results

| Model           |  K |   Recall |     NDCG |      MRR |      MAP | Hit Rate |
| --------------- | -: | -------: | -------: | -------: | -------: | -------: |
| TwoStageLearned | 10 | 0.001467 | 0.000775 | 0.000806 | 0.000416 | 0.002851 |
| TwoStageLearned | 20 | 0.003028 | 0.001252 | 0.001037 | 0.000519 | 0.006320 |
| TwoStageLearned | 40 | 0.005517 | 0.001879 | 0.001225 | 0.000608 | 0.011789 |
| TwoStageLearned | 80 | 0.009284 | 0.002686 | 0.001380 | 0.000676 | 0.020610 |

Candidate Recall@100 remained:

```text
0.010355
```

for every K because all final rankings used the same FAISS top-100 candidate set.

---

### Four-model comparison

| Model             |  K |   Recall |     NDCG |      MRR |      MAP | Hit Rate |
| ----------------- | -: | -------: | -------: | -------: | -------: | -------: |
| Popularity        | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALS               | 10 | 0.001172 | 0.000670 | 0.000800 | 0.000359 | 0.002632 |
| TwoStageHeuristic | 10 | 0.001158 | 0.000662 | 0.000791 | 0.000354 | 0.002613 |
| TwoStageLearned   | 10 | 0.001467 | 0.000775 | 0.000806 | 0.000416 | 0.002851 |
| Popularity        | 20 | 0.000007 | 0.000003 | 0.000002 | 0.000000 | 0.000034 |
| ALS               | 20 | 0.002250 | 0.001004 | 0.000972 | 0.000430 | 0.005211 |
| TwoStageHeuristic | 20 | 0.002241 | 0.000998 | 0.000964 | 0.000426 | 0.005196 |
| TwoStageLearned   | 20 | 0.003028 | 0.001252 | 0.001037 | 0.000519 | 0.006320 |
| Popularity        | 40 | 0.000779 | 0.000200 | 0.000067 | 0.000028 | 0.001786 |
| ALS               | 40 | 0.004535 | 0.001578 | 0.001148 | 0.000510 | 0.010319 |
| TwoStageHeuristic | 40 | 0.004513 | 0.001568 | 0.001138 | 0.000505 | 0.010246 |
| TwoStageLearned   | 40 | 0.005517 | 0.001879 | 0.001225 | 0.000608 | 0.011789 |
| Popularity        | 80 | 0.001092 | 0.000262 | 0.000077 | 0.000034 | 0.002394 |
| ALS               | 80 | 0.008415 | 0.002398 | 0.001299 | 0.000578 | 0.019053 |
| TwoStageHeuristic | 80 | 0.008390 | 0.002387 | 0.001289 | 0.000574 | 0.018999 |
| TwoStageLearned   | 80 | 0.009284 | 0.002686 | 0.001380 | 0.000676 | 0.020610 |

---

### Comparison with direct ALS

The learned reranker achieved higher Recall, NDCG, MRR, MAP, and Hit Rate than direct ALS at every evaluated cutoff.

Recall improvements relative to ALS were approximately:

```text
Recall@10: 25.2%
Recall@20: 34.6%
Recall@40: 21.7%
Recall@80: 10.3%
```

The largest relative Recall improvement occurred at:

```text
K = 20
```

where:

```text
ALS Recall@20:             0.002250
Learned Recall@20:         0.003028
Relative improvement:      approximately 34.6%
```

The learned reranker also consistently outperformed the heuristic reranker.

---

### Interpretation

The heuristic reranker used:

```text
0.99 × normalized ALS score
+
0.01 × normalized log-popularity
```

It performed slightly below direct ALS at each K.

This indicates that even a small fixed popularity contribution can move some relevant news items downward when static train-period popularity does not align with future dev-period clicks.

The learned reranker performed better because it learned how to combine:

```text
ALS compatibility
item popularity
history length
category affinity
subcategory affinity
```

from impression-level click labels instead of using manually selected weights.

The sample recommendation output also confirmed that the ranker changed the candidate ordering substantially. For example, some items originally retrieved near ranks 70, 85, or 100 were promoted into the final top-10.

---

### Retrieval limitation

The learned reranker can only reorder items already returned by FAISS.

It cannot recover a relevant item outside the top-100 candidate set.

The ratio:

```text
Recall@80 / Candidate Recall@100
=
0.009284 / 0.010355
≈ 0.897
```

shows that the learned reranker retained approximately 89.7% of the available candidate recall within its top-80 output.

This suggests that further improvements will increasingly depend on improving the retrieval stage, for example through:

```text
larger candidate sets
better ALS tuning
hybrid retrieval
content embeddings
two-tower retrieval
recency-aware retrieval
```

---

### Latency

```text
Feature-part construction:       169.956826 seconds
Ranker training:                  59.903855 seconds
FAISS index construction:          0.020937 seconds
FAISS retrieval:                  83.889672 seconds
Feature construction + reranking: 34.157593 seconds
End-to-end retrieval + reranking: 118.047265 seconds
```

Per-user serving latency:

```text
FAISS retrieval:          0.408151 ms/user
Feature + reranking:      0.166188 ms/user
End-to-end:               0.574339 ms/user
```

The learned-reranking stage added approximately:

```text
0.166 ms per user
```

after candidate retrieval.

The experiment therefore improved ranking quality while maintaining sub-millisecond average offline processing time per user.

---

### Generated Step 18b files

```text
data/processed/mindlarge/learned_reranker_candidates_sample.csv
data/processed/mindlarge/learned_reranker_top10.csv
data/processed/mindlarge/learned_reranker_evaluation.csv
data/processed/mindlarge/learned_reranker_model_comparison.csv
data/processed/mindlarge/learned_reranker_latency.json
```

Existing training artifacts:

```text
data/processed/mindlarge/learned_reranker_feature_summary.json
data/processed/mindlarge/learned_reranker_feature_mean.npy
data/processed/mindlarge/learned_reranker_feature_std.npy
data/processed/mindlarge/learned_reranker_training_history.csv
data/processed/mindlarge/learned_reranker_best.pt
data/processed/mindlarge/learned_reranker_features/train/
data/processed/mindlarge/learned_reranker_features/validation/
```

---

### Limitations

The sigmoid-transformed MLP output is a ranking-oriented score between zero and one. It should not be interpreted as a calibrated click-through probability.

The internal 90/10 validation split was used for ranker checkpoint selection. However, the ALS factors and popularity feature were previously constructed using the complete official train split. Therefore, the internal validation loss is useful for training diagnostics and early stopping, but it should not be presented as a completely isolated estimate of final ranking performance.

The official dev evaluation remains the primary model-comparison result because official dev click labels were not used to train the MLP or construct candidate features.

---

### Conclusion

The Full MIND learned-reranker experiment completed successfully.

The final system implements:

```text
Full MIND impression-level ranker training
→ memory-safe feature parts
→ PyTorch MLP
→ exact FAISS top-100 retrieval
→ learned candidate reranking
→ final top-K recommendations
→ five ranking metrics
→ four-model comparison
→ latency measurement
```

The learned reranker consistently outperformed Popularity, direct ALS, and the heuristic ALS-popularity reranker at K = 10, 20, 40, and 80.

Separating FAISS retrieval and PyTorch reranking into two independent processes resolved the macOS OpenMP conflict without using an unsafe duplicate-runtime workaround.
