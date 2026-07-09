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


