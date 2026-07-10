# Debiased News Recommendation via Low-Rank Matrix Factorization and Counterfactual Evaluation

This project builds a mathematically grounded news recommendation system based on low-rank matrix factorization, implicit feedback modeling, ranking-based evaluation, and counterfactual-style debiasing.

The project treats recommendation not simply as click prediction, but as learning latent user-item preference structure from sparse and biased implicit feedback data.

---

## Table of Contents

* [Project Motivation](#project-motivation)
* [Current Stage](#current-stage)
* [Mathematical Formulation](#mathematical-formulation)

  * [1. User-Item Interaction Matrix](#1-user-item-interaction-matrix)
  * [2. Low-Rank Latent Factor Model](#2-low-rank-latent-factor-model)
  * [3. Implicit Matrix Factorization Objective](#3-implicit-matrix-factorization-objective)
  * [4. Explicit Feedback versus Implicit Feedback](#4-explicit-feedback-versus-implicit-feedback)
* [Toy Low-Rank Recommender](#toy-low-rank-recommender)

  * [Truncated SVD Baseline](#truncated-svd-baseline)
  * [Why SVD Is Only a Baseline](#why-svd-is-only-a-baseline)
* [MIND-small Data Processing](#mind-small-data-processing)

  * [Interaction Tables](#interaction-tables)
  * [Train-Based ID Mappings](#train-based-id-mappings)
  * [Sparse Interaction Matrices](#sparse-interaction-matrices)
  * [Warm-Start Dev Ground Truth](#warm-start-dev-ground-truth)
* [Popularity Baseline](#popularity-baseline)
* [Implicit ALS Recommender](#implicit-als-recommender)

  * [ALS Model Configuration](#als-model-configuration)
  * [ALS Training Results](#als-training-results)
  * [ALS Recommendation Scores](#als-recommendation-scores)
* [Ranking Evaluation](#ranking-evaluation)

  * [Evaluation Protocol](#evaluation-protocol)
  * [Evaluated Users](#evaluated-users)
  * [Ranking Metrics](#ranking-metrics)
  * [Metric Sanity Checks](#metric-sanity-checks)
  * [Popularity Drift Diagnostic](#popularity-drift-diagnostic)
  * [Popularity versus ALS Results](#popularity-versus-als-results)
  * [Interpretation of the Results](#interpretation-of-the-results)
  * [Evaluation Limitations](#evaluation-limitations)
* [Full MIND Scale-Up](#full-mind-scale-up)

  * [Streaming Positive Interaction Processing](#streaming-positive-interaction-processing)
  * [Full MIND Train-Based ID Mappings](#full-mind-train-based-id-mappings)
  * [Full MIND Sparse Interaction Matrices](#full-mind-sparse-interaction-matrices)
  * [Full MIND ALS Training](#full-mind-als-training)
  * [Full MIND Popularity and ALS Recommendations](#full-mind-popularity-and-als-recommendations)
  * [Full MIND Popularity and ALS Evaluation](#full-mind-popularity-and-als-evaluation)
  * [Full MIND FAISS Two-Stage Pipeline](#full-mind-faiss-two-stage-pipeline)
  * [MIND-small versus Full MIND](#mind-small-versus-full-mind)
* [Generated Files](#generated-files)
* [Repository Structure](#repository-structure)
* [Main Files](#main-files)
* [How to Run](#how-to-run)
* [Requirements](#requirements)
* [Current Progress](#current-progress)
* [Next Steps](#next-steps)
* [Long-Term Project Roadmap](#long-term-project-roadmap)
* [Tech Stack](#tech-stack)
* [Project Goal](#project-goal)

---

## Project Motivation

News recommendation data is sparse and biased. A user only clicks articles that are exposed to them, so an unclicked article does not necessarily mean that the user dislikes it. It may simply mean that the user never saw the article.

The long-term goal of this project is to build a full recommendation pipeline:

```text
sparse and biased user-item interaction matrix
→ low-rank matrix factorization
→ top-K ranking evaluation
→ exposure bias / popularity bias analysis
→ two-stage retrieval and ranking
→ demo / API / optional Azure deployment
```

The project has moved from the toy low-rank example to a working MIND-small recommendation pipeline.

The current implementation includes:

```text
MIND-small data processing
→ sparse user-item matrix construction
→ Popularity baseline
→ implicit ALS
→ personalized top-K recommendation
→ whole-catalog ranking evaluation
```

The core collaborative-filtering pipeline has also been scaled from MIND-small to Full MIND:

```text
Full MIND behavior logs
→ streaming positive-interaction extraction
→ partitioned Parquet storage
→ train-based global ID mappings
→ binary CSR matrix construction
→ implicit ALS training
→ Popularity and ALS recommendation generation
→ whole-catalog ranking evaluation
→ FAISS top-100 candidate retrieval
→ heuristic reranking
→ final top-K recommendations
```

---

## Current Stage

The current version has completed:

* Project repository structure
* Mathematical problem formulation
* Toy user-item interaction matrix
* Hand-written truncated SVD implementation
* Low-rank score matrix reconstruction
* Top-K recommendation generation from reconstructed scores
* MIND-small raw file inspection
* Impression-log parsing and user-history parsing
* News metadata merge into train/dev interaction tables
* Train-based user/item integer index mappings
* Sparse train/dev user-item interaction matrices
* Static global Popularity baseline
* Implicit-feedback ALS training
* Personalized ALS top-K recommendation generation
* Recall@K
* NDCG@K
* MRR@K
* MAP@K
* Hit Rate@K
* Whole-catalog comparison between Popularity and ALS at multiple values of K
* Full MIND streaming behavior processing
* Full MIND positive interactions stored in partitioned Parquet files
* Full MIND train-based global user/item mappings
* Full MIND binary train/dev CSR matrices
* Full MIND implicit ALS training and saved latent factors
* Full MIND Popularity and ALS sample recommendations
* Full MIND whole-catalog Popularity versus ALS evaluation
* Full MIND FAISS `IndexFlatIP` candidate retrieval
* Full MIND top-100 unseen candidate generation
* Full MIND heuristic ALS-plus-popularity reranking
* Full MIND Popularity versus ALS versus TwoStage comparison
* Full MIND candidate-retrieval and reranking latency measurement

The current pipeline uses MIND-small and produces reproducible baseline results under a warm-start whole-catalog evaluation protocol.

The MIND-small two-stage pipeline uses:

```text
ALS user/item factors
→ FAISS candidate retrieval
→ top-100 candidates
→ candidate filtering
→ reranking
→ final top-10 recommendations
```

For Full MIND, Steps 11–17 now complete the pipeline from memory-safe behavior processing through recommendation generation, baseline evaluation, exact FAISS retrieval, heuristic reranking, and latency measurement. The next modeling step is a learned reranker trained from impression-level clicked and exposed-but-not-clicked candidates, followed by a controlled MIND-small versus Full MIND comparison.

---

## Mathematical Formulation

### 1. User-Item Interaction Matrix

Let there be (m) users and (n) news items. Define the observed user-item interaction matrix

$$
R \in {0,1}^{m\times n},
$$

where (R_{ui}=1) means user (u) clicked news item (i). However, in implicit feedback recommendation, (R_{ui}=0) does not necessarily mean that the user dislikes the item. It may simply mean that the user was never exposed to it.

Let

$$
E \in {0,1}^{m\times n}
$$

denote the exposure indicator matrix, and let

$$
Y \in {0,1}^{m\times n}
$$

be the latent preference matrix. Here, (E_{ui}=1) indicates that item (i) was exposed to user (u), and (Y_{ui}=1) indicates that user (u) would click or like item (i).

Then the observed user-item interaction matrix is generated by

$$
R = E \odot Y,
$$

where (\odot) denotes the Hadamard, or entry-wise, product.

The recommender system wants to learn the latent preference matrix (Y), but only observes the biased interaction matrix (R).

---

### 2. Low-Rank Latent Factor Model

Let

$$
S \in \mathbb{R}^{m\times n}
$$

be the predicted latent preference score matrix, where (S_{ui}) is the predicted preference score of user (u) for news item (i).

Instead of learning (S) as an arbitrary (m\times n) matrix, we assume that (S) has a low-rank latent factor structure:

$$
S = P Q^\top,
$$

where

$$
P \in \mathbb{R}^{m\times k},
\qquad
Q \in \mathbb{R}^{n\times k}.
$$

Then

$$
\operatorname{rank}(S)\leq k.
$$

For each user-item pair,

$$
S_{ui} = P^{(u)} \cdot Q^{(i)},
$$

where (P^{(u)}) is the (u)-th row of (P), (Q^{(i)}) is the (i)-th row of (Q), and both vectors lie in the same (k)-dimensional latent space.

The vector (P^{(u)}) represents the latent interests of user (u), while (Q^{(i)}) represents the latent features of news item (i).

We impose this low-rank structure because user preferences and news items are assumed to be governed by a relatively small number of latent factors. Therefore, the large sparse user-item matrix can be approximated through lower-dimensional user and item embeddings.

This low-rank factorization is a standard form of collaborative filtering: the model learns user and item representations from collective interaction patterns rather than relying only on explicit user features or item content.

---

### 3. Implicit Matrix Factorization Objective

The goal is to find user and item latent factor matrices

$$
P\in\mathbb{R}^{m\times k},
\qquad
Q\in\mathbb{R}^{n\times k}
$$

by minimizing the following regularized objective:

$$
\sum_{u,i}
c_{ui}
\left(
R_{ui}-P^{(u)}\cdot Q^{(i)}
\right)^2
+
\lambda
\left(
|P|_F^2+|Q|_F^2
\right).
$$

Each term in this objective has a specific interpretation.

The term

$$
\left(
R_{ui}-P^{(u)}\cdot Q^{(i)}
\right)^2
$$

measures the prediction error at the ((u,i))-th entry. It compares the observed interaction (R_{ui}) with the predicted preference score (P^{(u)}\cdot Q^{(i)}).

The coefficient (c_{ui}) is the confidence weight. A common choice is

$$
c_{ui}=1+\alpha R_{ui},
$$

where (\alpha>0).

If (R_{ui}=0), then

$$
c_{ui}=1,
$$

so the unclicked entry is treated as a low-confidence observation.

If (R_{ui}=1), then

$$
c_{ui}=1+\alpha,
$$

so the clicked entry receives a larger weight.

This reflects the idea that clicks are reliable positive signals, while non-clicks are weak and ambiguous signals.

Therefore, the model pays more attention to clicked interactions but still uses unclicked entries as weak information.

In implicit feedback recommendation, we sum over all user-item pairs rather than only the nonzero entries. A clicked entry (R_{ui}=1) is a strong positive signal, but an unclicked entry (R_{ui}=0) is not a strong negative signal. It may mean that the user disliked the item, but it may also mean that the user was never exposed to the item.

If we trained only on clicked entries, the model would learn which items should receive high scores, but it would not learn how to distinguish them from the large number of unclicked items.

By considering all user-item pairs, the model is encouraged to assign high scores to clicked items while keeping the scores of unclicked items relatively low, but only with low confidence.

The final term

$$
\lambda
\left(
|P|_F^2+|Q|_F^2
\right)
$$

is a regularization term.

The squared Frobenius norms (|P|_F^2) and (|Q|_F^2) measure the total squared magnitude of all entries in the user and item latent factor matrices.

This term penalizes overly large latent factors and helps prevent overfitting. Without regularization, the model might fit the training data too closely by learning very large or highly specialized embeddings. The parameter (\lambda) controls the strength of this penalty.

---

### 4. Explicit Feedback versus Implicit Feedback

This objective differs from the standard explicit-feedback matrix factorization objective.

In explicit-feedback settings, such as movie ratings, the loss is usually computed only over observed ratings, because missing ratings are truly unknown and should not be treated as zero.

In contrast, implicit-feedback data records user behavior such as clicks, views, or purchases. A clicked item provides a positive signal, while an unclicked item is ambiguous: it may indicate lack of interest, but it may also mean that the user was never exposed to the item.

Therefore, implicit matrix factorization considers all user-item pairs, but assigns different confidence weights to clicked and unclicked entries.

Clicked entries receive high confidence, while unclicked entries are treated as low-confidence observations rather than strong negatives.

---

## Toy Low-Rank Recommender

The project includes a small toy example to illustrate the mathematical idea behind low-rank recommendation.

The toy user-item matrix is:

```text
        Politics  Tech  Sports  Finance  Health  Travel
u1          1       1      0        0       0       0
u2          1       1      0        1       0       0
u3          0       0      1        0       1       0
u4          0       0      1        0       1       1
u5          1       0      0        1       0       0
```

The goal is to show how a sparse interaction matrix can be approximated by a low-rank score matrix, and how the reconstructed scores can be used to recommend unseen items.

---

### Truncated SVD Baseline

For the toy example, we approximate the user-item matrix (R) using truncated SVD.

Given the singular value decomposition

$$
R = U\Sigma V^\top,
$$

the rank-(k) approximation is

$$
R_k = U_k \Sigma_k V_k^\top.
$$

By the Eckart-Young theorem,

$$
R_k
===

\arg\min_{\operatorname{rank}(S)\leq k}
|R-S|_F^2.
$$

Therefore, truncated SVD gives the best rank-(k) approximation to (R) under the Frobenius norm.

In this project stage, the reconstructed matrix (R_k) is used as a score matrix. For each user, already-clicked items are removed, and the remaining items are ranked by their reconstructed scores.

---

### Why SVD Is Only a Baseline

Plain truncated SVD solves an unweighted and unregularized low-rank approximation problem:

$$
\min_{\operatorname{rank}(S)\leq k}
|R-S|_F^2.
$$

This is useful as a mathematical baseline, but real implicit-feedback data requires a more careful objective.

Plain SVD does not explicitly include confidence weights (c_{ui}), regularization (\lambda), exposure bias, or ranking-based loss.

It also treats all entries equally, while real implicit-feedback data contains strong positive signals and weak ambiguous non-clicks.

For real MIND-style news recommendation, the current project uses implicit ALS and will later add BPR and hybrid recommendation methods.

---

## MIND-small Data Processing

The MIND-small pipeline produces cleaned interaction tables and sparse user-item matrices for downstream recommendation models.

The raw behavior files contain:

```text
impression_id
user_id
time
history
impressions
```

The raw news files contain:

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

Each behavior row represents one impression event rather than one unique user. The same user may therefore appear in multiple rows.

### Interaction Tables

The `impressions` field is expanded into individual user-item rows.

For example:

```text
N12345-1 N23456-0 N34567-0
```

is interpreted as:

```text
N12345 was exposed and clicked.
N23456 was exposed but not clicked.
N34567 was exposed but not clicked.
```

The `history` field contains news items clicked before the current impression.

The processed tables preserve the distinction between:

```text
source = "history"
source = "impression"
```

The final cleaned interaction columns are:

```text
user_id
item_id
click
impression_id
time
source
category
subcategory
title
abstract
```

Processed interaction files:

```text
../data/processed/interactions_history_train.parquet
../data/processed/interactions_history_dev.parquet
../data/processed/train_with_news.parquet
../data/processed/dev_with_news.parquet
../data/processed/news.parquet
```

### Train-Based ID Mappings

Raw string identifiers are converted into integer matrix indices:

```text
user_id → user_idx
item_id → item_idx
```

Reverse mappings are also saved:

```text
user_idx → user_id
item_idx → item_id
```

Mapping files:

```text
../data/processed/user_idx_map.json
../data/processed/item_idx_map.json
../data/processed/idx_user_map.json
../data/processed/idx_item_map.json
```

The mappings are created from training data only.

This ensures that the training matrix defines the common row and column spaces used by classical matrix-based models.

### Sparse Interaction Matrices

The pipeline constructs binary SciPy CSR matrices.

The train matrix contains unique positive train interactions from:

```text
source = "history"
source = "impression" with click = 1
```

The dev matrix is used only as validation ground truth.

Sparse matrix files:

```text
../data/processed/train_interactions.npz
../data/processed/dev_interactions.npz
```

Latest sparse matrix results:

```text
train matrix shape: (50000, 51282)
dev matrix shape:   (50000, 51282)

train nnz: 1148447
dev nnz:     10277
```

The matrices contain:

```text
50,000 train-known users
51,282 train-known news items
1,148,447 unique positive train user-item pairs
10,277 final positive dev user-item pairs
```

### Warm-Start Dev Ground Truth

The dev matrix is constructed using three main filtering decisions.

#### 1. Use clicked dev impressions

The dev matrix uses only:

```text
click = 1
source = "impression"
```

Dev history rows are excluded because they describe previous behavior rather than the future interaction targets the model should recover.

#### 2. Apply warm-start filtering

Only users and items present in the train-based mappings are retained.

Matrix-factorization models such as ALS cannot directly score completely new users or items because they have no learned latent factors.

#### 3. Remove train-seen user-item pairs

A dev positive pair is removed if the same user-item pair already appears in the train positives.

Recommendation generation filters items already clicked in train. Keeping those same items as dev targets would ask the model to recover items that it is explicitly prohibited from recommending.

The filtering results are:

```text
Dev clicked impression pairs before filtering: 110,745
Dev warm-start pairs after user/item filtering: 10,314
Dev positive pairs already seen in train:            37
Final dev positive pairs:                        10,277
```

---

## Popularity Baseline

The project implements a static global Popularity baseline.

For each item (i), its popularity score is:

$$
\operatorname{popularity}(i)
============================

\sum_u R_{ui}.
$$

Because the train matrix is binary and duplicate user-item pairs were removed, this score represents the number of unique training users who clicked the item.

The recommendation procedure is:

```text
1. Calculate item popularity from train only.
2. Sort all items from most popular to least popular.
3. Retrieve the target user's train-seen items.
4. Remove those items from the global ranking.
5. Return the top-K remaining items.
```

The model is non-personalized because all users share the same global popularity ordering before filtering.

Popularity results:

```text
Train matrix shape:              (50000, 51282)
Train matrix nnz:                1148447
Items with at least one click:     39865
Items with zero train clicks:      11417
Maximum popularity score:           4747
```

The most popular item was clicked by 4,747 unique training users.

Generated files:

```text
../data/processed/popularity_scores.npy
../data/processed/popularity_ranking.npy
```

`popularity_scores.npy` stores:

```text
popularity_scores[item_idx]
=
number of unique training users who clicked the item
```

`popularity_ranking.npy` stores all item indices in descending popularity order.

---

## Implicit ALS Recommender

ALS is the first personalized collaborative-filtering model implemented on the MIND-small sparse matrix.

The model learns:

$$
P\in\mathbb{R}^{m\times k},
\qquad
Q\in\mathbb{R}^{n\times k},
$$

where (P) contains user factors and (Q) contains item factors.

The predicted score is:

$$
s(u,i)=p_u^\top q_i.
$$

ALS alternates between:

```text
1. Fixing item factors and updating user factors.
2. Fixing user factors and updating item factors.
```

Although the full objective is jointly non-convex in (P) and (Q), each update becomes a regularized least-squares problem when the other factor matrix is fixed.

### ALS Model Configuration

The model was trained using `AlternatingLeastSquares` from the Python `implicit` library.

```text
factors:        64
regularization: 0.1
alpha:          40.0
iterations:     15
random_state:   42
use_gpu:        False
```

Interpretation:

```text
factors = 64:
Each user and item is represented by a 64-dimensional latent vector.

regularization = 0.1:
Penalizes large factor values and reduces overfitting.

alpha = 40.0:
Increases the confidence assigned to observed positive interactions.

iterations = 15:
ALS performs 15 alternating update rounds.
```

The model stops after 15 rounds because the number of iterations was specified manually. Automatic early stopping was not used.

### ALS Training Results

Training input:

```text
Train matrix shape: (50000, 51282)
Train matrix nnz:   1148447
Matrix dtype:       float32
```

Learned factor matrices:

```text
User factors shape: (50000, 64)
Item factors shape: (51282, 64)
```

This confirms that the model learned:

```text
one 64-dimensional factor vector for each user
one 64-dimensional factor vector for each news item
```

Generated files:

```text
../data/processed/als_model.npz
../data/processed/als_user_factors.npy
../data/processed/als_item_factors.npy
```

The separately saved factor matrices will be reused for FAISS candidate retrieval.

### ALS Recommendation Scores

The model generates recommendations using:

$$
s(u,i)=p_u^\top q_i.
$$

Items already clicked by the target user in train are filtered out.

The ALS score is a latent-factor ranking score.

It is not a calibrated click probability.

For example:

```text
ALS score = 0.91
```

does not mean:

```text
91% probability of clicking
```

It means that the item received a higher latent compatibility score than lower-ranked eligible items.

For sample user `U100`, the model generated personalized recommendations different from the static Popularity ranking, showing that ALS uses user-specific collaborative patterns.

---

## Ranking Evaluation

The project evaluates Popularity and ALS using ranking metrics rather than RMSE.

The goal is not to predict a numeric rating. The goal is to rank relevant future news items near the top of a recommendation list.

### Evaluation Protocol

The models are trained using:

```text
../data/processed/train_interactions.npz
```

Validation ground truth is loaded from:

```text
../data/processed/dev_interactions.npz
```

For every evaluated user:

```text
1. Generate top-K recommendations.
2. Use all 51,282 train-known items as the candidate universe.
3. Remove items already clicked by the user in train.
4. Compare the recommendations with the user's valid dev clicked items.
5. Compute ranking metrics.
```

This is a warm-start whole-catalog evaluation.

It is not impression-level evaluation.

The model must retrieve relevant articles from the complete train-known catalog rather than rank only the articles contained in one MIND impression.

### Evaluated Users

The sparse matrices contain 50,000 train-known user rows, but only 5,109 users are included in the ranking evaluation.

A user must have at least one valid dev positive after:

```text
1. keeping clicked dev impression items;
2. applying warm-start user/item filtering;
3. removing dev positive pairs already observed in train;
4. keeping users with non-empty remaining dev ground truth.
```

Final evaluation statistics:

```text
Evaluated users:          5,109
Valid dev positive pairs: 10,277
Average positives/user:   approximately 2.01
```

The other matrix rows remain valid train users, but they have no remaining evaluation target in the final dev matrix.

### Ranking Metrics

#### Recall@K

For user (u):

$$
\operatorname{Recall@K}(u)
==========================

\frac{
|\operatorname{Recommended}_u@K
\cap
\operatorname{Relevant}_u|
}{
|\operatorname{Relevant}_u|
}.
$$

Recall@K measures the fraction of relevant dev items recovered in the top-K recommendation list.

#### NDCG@K

NDCG@K rewards relevant items more when they appear near the top of the recommendation list.

For binary relevance:

$$
\operatorname{DCG@K}
====================

\sum_{r=1}^{K}
\frac{\operatorname{rel}_r}{\log_2(r+1)}.
$$

NDCG divides DCG by the ideal DCG.

A perfect ranking receives:

$$
\operatorname{NDCG@K}=1.
$$

#### MRR@K

For one user:

$$
\operatorname{RR@K}
===================

\frac{1}{
\text{rank of the first relevant recommended item}
}.
$$

If there is no relevant item in the top-K list, Reciprocal Rank is zero.

MRR averages Reciprocal Rank across users.

#### MAP@K

Average Precision computes precision at each rank containing a relevant item.

MAP@K averages Average Precision across users.

It rewards relevant items that appear early and maintains high precision at relevant ranks.

#### Hit Rate@K

For one user:

$$
\operatorname{Hit@K}
====================

\begin{cases}
1, & \text{if at least one relevant item appears in top-K},\
0, & \text{otherwise}.
\end{cases}
$$

Hit Rate@K is the fraction of users receiving at least one relevant recommendation.

### Metric Sanity Checks

The metric functions were tested using manual examples.

```text
Perfect recommendation:
Recall = 1
NDCG = 1
MRR = 1
MAP = 1
Hit Rate = 1

No-hit recommendation:
Recall = 0
NDCG = 0
MRR = 0
MAP = 0
Hit Rate = 0
```

The implementation passed these tests:

```text
Metric sanity checks passed.
```

### Popularity Drift Diagnostic

Before evaluating the models, the project measures how many highly popular train items appear anywhere in the dev positive-item set.

```text
Popularity top-10 items appearing in dev positives:    0
Popularity top-100 items appearing in dev positives:   4
Popularity top-1000 items appearing in dev positives: 42
```

None of the ten most popular training items appeared among the valid dev positives.

Only four of the top 100 and 42 of the top 1,000 appeared in the dev positives.

This result is consistent with strong temporal drift in news popularity:

```text
An article that was popular during the training period may no longer
be relevant or actively clicked during the development period.
```

### Popularity versus ALS Results

The models were evaluated at:

```text
K = 10
K = 20
K = 40
K = 80
```

Results:

| Model      |  K |   Recall |     NDCG |      MRR |      MAP | Hit Rate |
| ---------- | -: | -------: | -------: | -------: | -------: | -------: |
| Popularity | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALS        | 10 | 0.001783 | 0.000916 | 0.000778 | 0.000546 | 0.003132 |
| Popularity | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALS        | 20 | 0.003258 | 0.001343 | 0.000961 | 0.000651 | 0.005676 |
| Popularity | 40 | 0.000525 | 0.000138 | 0.000051 | 0.000019 | 0.001370 |
| ALS        | 40 | 0.005296 | 0.001831 | 0.001097 | 0.000718 | 0.009787 |
| Popularity | 80 | 0.001242 | 0.000271 | 0.000069 | 0.000032 | 0.002349 |
| ALS        | 80 | 0.009650 | 0.002712 | 0.001243 | 0.000797 | 0.018203 |

The results are saved to:

```text
../data/processed/ranking_evaluation.csv
```

### Interpretation of the Results

#### ALS consistently outperformed static Popularity

ALS achieved higher Recall, NDCG, MRR, MAP, and Hit Rate at every tested value of K.

At (K=80):

```text
ALS Recall@80:        0.009650
Popularity Recall@80: 0.001242
```

ALS achieved approximately:

```text
0.009650 / 0.001242 ≈ 7.8
```

times the Recall of Popularity.

For Hit Rate:

```text
ALS HitRate@80:        0.018203
Popularity HitRate@80: 0.002349
```

ALS achieved approximately 7.7 times the Hit Rate of Popularity.

The appropriate conclusion is:

> ALS consistently outperformed the static global Popularity baseline under the current warm-start whole-catalog evaluation protocol.

This conclusion applies to the implemented static Popularity baseline. It does not imply that ALS outperforms every possible time-aware or trending-popularity model.

#### Why Popularity was zero at K = 10 and K = 20

Popularity produced zero values for every metric at (K=10) and (K=20).

This does not indicate an implementation error.

The train-period Popularity ranking had almost no overlap with valid dev positive items.

Because users receive almost the same global ranking, the short recommendation lists did not contain future clicked articles.

#### Why Popularity became nonzero at K = 40

Popularity began producing hits at (K=40):

```text
Recall@40:  0.000525
HitRate@40: 0.001370
```

There were four dev-positive items among the global top-100 train-popular items.

In addition, train-seen items are filtered separately for every user. If a user already clicked some highly popular items, lower-ranked global items move into that user's final top-K list.

This can allow one of the few overlapping items to enter a user's recommendation list.

#### Increasing K increases coverage

ALS Hit Rate increased as K became larger:

```text
HitRate@10: 0.003132
HitRate@20: 0.005676
HitRate@40: 0.009787
HitRate@80: 0.018203
```

With 5,109 evaluated users, this corresponds approximately to:

```text
K = 10: 16 users with at least one hit
K = 20: 29 users with at least one hit
K = 40: 50 users with at least one hit
K = 80: 93 users with at least one hit
```

This increase is expected because a longer list provides more opportunities to contain a relevant item.

Metric values across different K values do not represent the same task difficulty.

The correct comparisons are:

```text
Popularity@10 versus ALS@10
Popularity@20 versus ALS@20
Popularity@40 versus ALS@40
Popularity@80 versus ALS@80
```

#### MRR and MAP increased more slowly

For ALS:

```text
MRR@10: 0.000778
MRR@80: 0.001243

MAP@10: 0.000546
MAP@80: 0.000797
```

Recall and Hit Rate increased more quickly than MRR and MAP.

This indicates that many additional hits found at larger K values appeared relatively far down the recommendation list.

Therefore:

```text
ALS has some useful whole-catalog retrieval ability,
but ranking relevant items near the top remains difficult.
```

#### Absolute performance remains low

Although ALS substantially outperformed Popularity, the absolute metrics remain low.

The strongest tested ALS results were:

```text
Recall@80:  0.009650
HitRate@80: 0.018203
```

Even when returning 80 recommendations, approximately 1.82% of evaluated users received at least one relevant item.

This does not mean that the ALS implementation failed.

The model must select a small number of relevant articles from more than 51,000 possible items.

Factors contributing to the difficulty include:

```text
1. Whole-catalog retrieval over 51,282 items.
2. Rapid temporal drift in news relevance.
3. Short article lifetimes.
4. Sparse user-item interactions.
5. Limited histories for some users.
6. No article recency feature.
7. No title, abstract, category, or subcategory features.
8. Untuned ALS hyperparameters.
9. An ALS objective that does not directly optimize Recall or NDCG.
```

### Evaluation Limitations

The current results should be interpreted within the following boundaries.

#### Whole-catalog rather than impression-level evaluation

The current candidate set contains all 51,282 train-known items.

A MIND impression usually contains a much smaller candidate set.

Therefore, the current results should not be directly compared with published impression-level MIND benchmark scores.

The two evaluation tasks measure different capabilities:

```text
Whole-catalog evaluation:
Tests candidate retrieval from the complete item universe.

Impression-level evaluation:
Tests ranking within a provided exposure candidate set.
```

#### Warm-start only

Dev-only users and dev-only items are excluded.

The current ALS model cannot directly score entities without learned training factors.

Cold-start recommendation will require content features, metadata, text embeddings, or fallback strategies.

#### Static Popularity baseline

The baseline aggregates popularity across the full training period.

Stronger news-specific popularity baselines could include:

```text
recent-window popularity
time-decayed popularity
trending popularity
category-specific popularity
```

#### No formal statistical significance test yet

ALS achieved higher observed aggregate metrics, but confidence intervals and paired statistical significance tests have not yet been computed.

A future analysis can use user-level bootstrap confidence intervals or paired permutation tests.

---


## Full MIND Scale-Up

After validating the classical recommendation pipeline on MIND-small, the project scaled the same interaction semantics and ALS configuration to Full MIND.

The Full MIND extension keeps the same modeling choices:

```text
Train positives:
history clicks + clicked train impressions

Dev ground truth:
clicked dev impressions only
→ keep train-known users/items
→ remove train-seen user-item pairs
```

The main difference is engineering scale. Full MIND is processed with streaming, partitioned storage, and batched sparse-matrix construction instead of loading one complete expanded interaction table into memory.

### Streaming Positive Interaction Processing

`notebooks/11_parse_mind_large_streaming.py` reads the Full MIND behavior files line by line.

Approximately 500,000 extracted interactions are buffered at a time and written to one Parquet part. Global user-item deduplication is intentionally delayed until sparse matrix construction so that streaming memory usage remains bounded.

Train results:

```text
Raw behavior rows:                 2,232,748
History interaction occurrences: 73,629,868
Clicked impression occurrences:   3,383,656
Total positive rows written:     77,013,524
Parquet part files:                      155
Malformed behavior rows:                    0
Malformed impression tokens:                0
Processing time:                  30.23 seconds
```

Dev results:

```text
Raw behavior rows:               376,471
Clicked impression occurrences: 574,845
Total positive rows written:     574,845
Parquet part files:                    2
Malformed behavior rows:                0
Malformed impression tokens:            0
Processing time:                1.99 seconds
```

The 77,013,524 train rows are positive interaction occurrences, not unique matrix entries. Repeated history and click occurrences are consolidated later into binary user-item pairs.

### Full MIND Train-Based ID Mappings

`notebooks/12_build_mind_large_mappings.py` creates one deterministic global mapping for all Full MIND partitions.

The mapping universes are defined directly from the raw train files:

```text
MINDlarge_train/behaviors.tsv → all train users
MINDlarge_train/news.tsv      → all train news items
```

The IDs are sorted before integer indices are assigned. Therefore, rerunning the script on the same input produces the same mapping.

Results:

```text
Train behavior rows:  2,232,748
Unique train users:     711,222
Train news rows:        101,527
Unique train items:     101,527
Matrix index space: (711,222, 101,527)
Processing time: approximately 3.2 seconds
```

A single global mapping ensures that the same raw user or item ID has the same matrix index across all 155 train partitions.

### Full MIND Sparse Interaction Matrices

`notebooks/13_build_mind_large_sparse_matrix.py` reads the partitioned Parquet files in batches, maps raw IDs to integer coordinates, creates local binary sparse matrices, and merges them into final CSR matrices.

Repeated user-item coordinates are consolidated so that:

```text
R[user_idx, item_idx] = 1
```

regardless of how many times the same positive pair appears in the raw behavior logs.

Train results:

```text
Parquet parts:          155
Raw positive rows:      77,013,524
Matrix shape:           (711,222, 101,527)
Unique positive pairs:  16,532,504
Density:                0.0002289559
```

Dev results:

```text
Parquet parts:             2
Raw clicked rows:          574,845
Warm-start rows:           463,600
Cold-start users:           39,212
Cold-start items:            1,359
Train-seen pairs removed:    1,702
Matrix shape:              (711,222, 101,527)
Final positive pairs:       459,068
Density:                   0.0000063576
```

Sparse matrix construction took 56.62 seconds on the local machine.

The theoretical dense matrix contains more than 72 billion entries, so dense storage would be impractical. CSR stores only the nonzero coordinates and supports efficient user-row access for ALS training and recommendation filtering.

### Full MIND ALS Training

`notebooks/14_train_mind_large_als.py` trains implicit ALS on the Full MIND binary train matrix.

The configuration is intentionally kept the same as the MIND-small baseline:

```text
factors:        64
regularization: 0.1
alpha:          40.0
iterations:     15
random_state:   42
use_gpu:        False
```

Training results:

```text
Train matrix shape:   (711,222, 101,527)
Train matrix nnz:     16,532,504
Training time:        88.91 seconds
Average time/round:   approximately 5.91 seconds

User factors shape:   (711,222, 64)
Item factors shape:   (101,527, 64)
Factor dtype:         float32
```

The script saves the complete ALS model and the factor matrices separately. Step 14 performs model fitting only; sample top-K recommendation, Popularity comparison, ranking evaluation, and FAISS retrieval are intentionally separated into later steps.


### Full MIND Popularity and ALS Recommendations

`notebooks/15_mind_large_recommendations.py` verifies that the Full MIND matrices, mappings, metadata, and trained ALS factors can produce valid readable recommendations.

The script performs the following operations:

```text
1. Load the Full MIND binary train interaction matrix.
2. Load the saved ALS user and item factors.
3. Compute item popularity from train only.
4. Save the global popularity scores and ranking.
5. Select a train-known user with nonempty history.
6. Generate Popularity top-10 unseen recommendations.
7. Generate ALS top-10 unseen recommendations.
8. Filter all train-seen items.
9. Verify ALS scores using direct user-item factor dot products.
10. Map matrix indices back to MIND user IDs, news IDs, and titles.
```

The Full MIND popularity score is:

$$
\operatorname{popularity}(i)=\sum_u R_{ui}.
$$

Because the train matrix is binary, this is the number of distinct train users who clicked item $i$, not the number of repeated click events.

The ALS score is:

$$
s(u,i)=p_u^\top q_i.
$$

It is a latent ranking score rather than a calibrated click probability.

Generated files:

```text
data/processed/mindlarge/popularity_scores.npy
data/processed/mindlarge/popularity_ranking.npy
data/processed/mindlarge/popularity_sample_top10.csv
data/processed/mindlarge/als_sample_top10.csv
```

The sample outputs confirm that both models return ten unique unseen articles and that the internal item indices can be converted into readable news IDs and titles.

### Full MIND Popularity and ALS Evaluation

`notebooks/16_mind_large_ranking_evaluation.py` evaluates static Popularity and implicit ALS under the same warm-start whole-catalog protocol.

Evaluation setup:

```text
Train-known users:        711,222
Train-known items:        101,527
Valid dev positive pairs: 459,068
Evaluated users:          205,536
Candidate universe:       all 101,527 train-known items
K values:                 10, 20, 40, 80
```

For every evaluated user, both models rank train-known items, remove items already clicked in train, and compare the resulting recommendations with the user's valid dev clicked impressions.

The evaluation uses:

```text
Recall@K
NDCG@K
MRR@K
MAP@K
Hit Rate@K
```

Results:

| Model | K | Recall | NDCG | MRR | MAP | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALS | 10 | 0.001172 | 0.000670 | 0.000800 | 0.000359 | 0.002632 |
| Popularity | 20 | 0.000007 | 0.000003 | 0.000002 | 0.000000 | 0.000034 |
| ALS | 20 | 0.002250 | 0.001004 | 0.000972 | 0.000430 | 0.005211 |
| Popularity | 40 | 0.000779 | 0.000200 | 0.000067 | 0.000028 | 0.001786 |
| ALS | 40 | 0.004535 | 0.001578 | 0.001148 | 0.000510 | 0.010319 |
| Popularity | 80 | 0.001092 | 0.000262 | 0.000077 | 0.000034 | 0.002394 |
| ALS | 80 | 0.008415 | 0.002398 | 0.001299 | 0.000578 | 0.019053 |

ALS outperformed static Popularity at every tested value of $K$ and on every metric. At $K=80$, ALS achieved approximately 7.7 times the Recall of Popularity:

```text
0.008415 / 0.001092 ≈ 7.7
```

The low absolute metric values reflect the difficulty of retrieving a small number of future clicked articles from a catalog of more than 100,000 items. The results also show that static train-period popularity is weak under rapid news temporal drift.

Generated file:

```text
data/processed/mindlarge/ranking_evaluation.csv
```

### Full MIND FAISS Two-Stage Pipeline

`notebooks/17_mind_large_two_stage_pipeline.py` scales the ALS–FAISS–reranking architecture from MIND-small to Full MIND.

The implemented pipeline is:

```text
ALS user factor
→ FAISS IndexFlatIP search
→ retrieve extra items before filtering
→ remove train-seen items
→ retain exactly 100 unique unseen candidates
→ recompute exact ALS candidate scores
→ add normalized log-popularity
→ heuristic reranking
→ evaluate final top-10, top-20, top-40, and top-80
```

#### Exact FAISS retrieval

The item-factor matrix is indexed with:

```python
faiss.IndexFlatIP
```

This is an exact maximum-inner-product index. It matches the ALS scoring function because both use:

$$
s(u,i)=p_u^\top q_i.
$$

The validation result was:

```text
Maximum FAISS score error: 5.960464477539063e-08
```

This difference is negligible float32 numerical error. The pipeline was also tested with:

```text
ALS weight:        1.0
Popularity weight: 0.0
```

Under this setting, the two-stage system reproduced the direct ALS metrics at all four values of $K$. This sanity check confirms that FAISS retrieval, train-seen filtering, and candidate construction do not reduce ALS ranking quality.

#### Candidate retrieval

For each evaluated user, the system retains exactly 100 unique unseen candidates.

```text
Candidate Recall@100: 0.010355
```

Candidate Recall@100 measures the fraction of relevant dev items contained anywhere in the retrieval set before final reranking. It is an upper bound on final Recall for a ranker restricted to the same 100 candidates.

#### Heuristic reranker

The evaluated heuristic configuration was:

```text
ALS weight:        0.99
Popularity weight: 0.01
```

For each user's candidate set, the two features are computed as:

```text
normalized ALS score
normalized log(1 + popularity)
```

The final score is:

$$
\operatorname{score}(u,i)
=
0.99\,\widetilde{s}_{\mathrm{ALS}}(u,i)
+
0.01\,\widetilde{\log(1+\operatorname{popularity}(i))}.
$$

The logarithmic transformation compresses the highly skewed popularity distribution, reduces the influence of extreme outliers, preserves popularity ordering, and safely handles zero-popularity items. Both features are then min-max normalized within the user's 100-item candidate set so they can be combined on comparable scales.

#### Popularity versus ALS versus TwoStage

| Model | K | Recall | NDCG | MRR | MAP | Hit Rate | Candidate Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — |
| ALS | 10 | 0.001172 | 0.000670 | 0.000800 | 0.000359 | 0.002632 | — |
| TwoStageHeuristic | 10 | 0.001158 | 0.000662 | 0.000791 | 0.000354 | 0.002613 | 0.010355 |
| Popularity | 20 | 0.000007 | 0.000003 | 0.000002 | 0.000000 | 0.000034 | — |
| ALS | 20 | 0.002250 | 0.001004 | 0.000972 | 0.000430 | 0.005211 | — |
| TwoStageHeuristic | 20 | 0.002241 | 0.000998 | 0.000964 | 0.000426 | 0.005196 | 0.010355 |
| Popularity | 40 | 0.000779 | 0.000200 | 0.000067 | 0.000028 | 0.001786 | — |
| ALS | 40 | 0.004535 | 0.001578 | 0.001148 | 0.000510 | 0.010319 | — |
| TwoStageHeuristic | 40 | 0.004513 | 0.001568 | 0.001138 | 0.000505 | 0.010246 | 0.010355 |
| Popularity | 80 | 0.001092 | 0.000262 | 0.000077 | 0.000034 | 0.002394 | — |
| ALS | 80 | 0.008415 | 0.002398 | 0.001299 | 0.000578 | 0.019053 | — |
| TwoStageHeuristic | 80 | 0.008390 | 0.002387 | 0.001289 | 0.000574 | 0.018999 | 0.010355 |

The heuristic reranker performed slightly below direct ALS at every tested value of $K$. Because the ALS-only sanity check reproduced direct ALS exactly, the difference is attributable to the static popularity feature rather than to FAISS retrieval.

A 1% popularity contribution can still change items near ranking boundaries after candidate-level normalization, such as ranks 10 and 11 or ranks 80 and 81. In this temporal news setting, historical cumulative popularity can promote articles that were popular during train but are stale during dev.

Therefore, the current result is:

> Exact FAISS retrieval preserves ALS quality, but adding static log-popularity does not improve the Full MIND ranking and slightly reduces all measured metrics.

#### Latency

```text
FAISS index build time:          0.0144 seconds
Candidate retrieval time:       42.4406 seconds
Candidate retrieval latency:     0.2065 ms/user
Filtering and reranking time:    16.3308 seconds
Reranking latency:                0.0795 ms/user
End-to-end evaluation time:      80.8330 seconds
End-to-end latency:               0.3933 ms/user
```

The end-to-end value includes Python control flow and offline metric calculation, so it should not be interpreted as pure online serving latency. Nevertheless, the experiment confirms that exact candidate retrieval and lightweight heuristic reranking are computationally inexpensive at the current scale.

Generated files:

```text
data/processed/mindlarge/two_stage_candidates_sample.csv
data/processed/mindlarge/two_stage_top10.csv
data/processed/mindlarge/two_stage_evaluation.csv
data/processed/mindlarge/two_stage_model_comparison.csv
data/processed/mindlarge/two_stage_latency.json
```

### MIND-small versus Full MIND

| Statistic | MIND-small | Full MIND |
| --- | ---: | ---: |
| Train-known users | 50,000 | 711,222 |
| Train-known items | 51,282 | 101,527 |
| Unique train positive pairs | 1,148,447 | 16,532,504 |
| Final warm-start dev pairs | 10,277 | 459,068 |
| ALS user-factor shape | (50,000, 64) | (711,222, 64) |
| ALS item-factor shape | (51,282, 64) | (101,527, 64) |
| ALS training time | approximately 5 seconds | 88.91 seconds |
| Processing strategy | complete DataFrames | streaming and partitioned batches |

The mathematical model is unchanged. The Full MIND work demonstrates that the same train-defined mappings, binary implicit-feedback matrix, warm-start dev protocol, and ALS factorization can be scaled to a substantially larger dataset.

Full MIND ranking metrics have now been computed with the same warm-start whole-catalog definitions and the same values of $K$ used for MIND-small. However, direct metric comparison still requires care because the Full MIND catalog, evaluated-user population, and number of validation interactions are substantially larger. Step 19 will summarize the scale, runtime, and model-quality differences under a single controlled comparison.

---

## Generated Files

### Processed interaction tables

```text
../data/processed/interactions_history_train.parquet
../data/processed/interactions_history_dev.parquet
../data/processed/train_with_news.parquet
../data/processed/dev_with_news.parquet
../data/processed/news.parquet
```

### ID mapping files

```text
../data/processed/user_idx_map.json
../data/processed/item_idx_map.json
../data/processed/idx_user_map.json
../data/processed/idx_item_map.json
```

### Sparse matrix files

```text
../data/processed/train_interactions.npz
../data/processed/dev_interactions.npz
```

### Popularity files

```text
../data/processed/popularity_scores.npy
../data/processed/popularity_ranking.npy
```

### ALS files

```text
../data/processed/als_model.npz
../data/processed/als_user_factors.npy
../data/processed/als_item_factors.npy
```

### Ranking evaluation file

```text
../data/processed/ranking_evaluation.csv
```

### Full MIND partitioned interaction files

```text
../data/processed/mindlarge/train_positive_interactions/part-*.parquet
../data/processed/mindlarge/dev_clicked_impressions/part-*.parquet
../data/processed/mindlarge/11_streaming_parse_summary.json
```

### Full MIND mapping files

```text
../data/processed/mindlarge/user_idx_map.json
../data/processed/mindlarge/item_idx_map.json
../data/processed/mindlarge/idx_user_map.json
../data/processed/mindlarge/idx_item_map.json
```

### Full MIND sparse matrix files

```text
../data/processed/mindlarge/train_interactions.npz
../data/processed/mindlarge/dev_interactions.npz
```

### Full MIND ALS files

```text
../data/processed/mindlarge/als_model.npz
../data/processed/mindlarge/als_user_factors.npy
../data/processed/mindlarge/als_item_factors.npy
```

### Full MIND recommendation and Popularity files

```text
../data/processed/mindlarge/popularity_scores.npy
../data/processed/mindlarge/popularity_ranking.npy
../data/processed/mindlarge/popularity_sample_top10.csv
../data/processed/mindlarge/als_sample_top10.csv
```

### Full MIND ranking evaluation file

```text
../data/processed/mindlarge/ranking_evaluation.csv
```

### Full MIND two-stage files

```text
../data/processed/mindlarge/two_stage_candidates_sample.csv
../data/processed/mindlarge/two_stage_top10.csv
../data/processed/mindlarge/two_stage_evaluation.csv
../data/processed/mindlarge/two_stage_model_comparison.csv
../data/processed/mindlarge/two_stage_latency.json
```

---

## Repository Structure

```text
recsys-news-debiasing/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── MINDsmall_train/
│   │   ├── MINDsmall_dev/
│   │   ├── MINDlarge_train/
│   │   └── MINDlarge_dev/
│   └── processed/
│       └── mindlarge/
├── docs/
│   ├── 01_problem_formulation.md
│   ├── decisions.md
│   ├── experiments.md
│   └── pitfalls.md
├── notebooks/
│   ├── 01_low_rank_recommender_toy_example.ipynb
│   ├── 02_inspect_raw_mind_small.ipynb
│   ├── 03_parse_impression_small.ipynb
│   ├── 04_0_check_parquet.py
│   ├── 04_merge_news_metadata.py
│   ├── 05_user_item_index_mapping.py
│   ├── 06_user_item_interaction_matrix.py
│   ├── 07_popularity_baseline.py
│   ├── 08_train_als.py
│   ├── 09_ranking_evaluation.py
│   ├── 10_two_stage_pipeline.py
│   ├── 11_parse_mind_large_streaming.py
│   ├── 12_build_mind_large_mappings.py
│   ├── 13_build_mind_large_sparse_matrix.py
│   ├── 14_train_mind_large_als.py
│   ├── 15_mind_large_recommendations.py
│   ├── 16_mind_large_ranking_evaluation.py
│   └── 17_mind_large_two_stage_pipeline.py
└── src/
    ├── data/
    ├── models/
    ├── evaluation/
    ├── retrieval/
    └── ranking/
```

---

## Main Files

* `README.md`: project overview, mathematical formulation, current results, and progress tracking
* `docs/01_problem_formulation.md`: detailed mathematical problem formulation
* `docs/experiments.md`: experiment logs, outputs, and result interpretation
* `docs/decisions.md`: design decisions and evaluation protocol choices
* `docs/pitfalls.md`: implementation pitfalls and fixes
* `notebooks/01_low_rank_recommender_toy_example.ipynb`: toy low-rank recommender with hand-written truncated SVD
* `notebooks/02_inspect_raw_mind_small.ipynb`: raw MIND-small data inspection
* `notebooks/03_parse_impression_small.ipynb`: impression and history parsing
* `notebooks/04_merge_news_metadata.py`: interaction and news metadata merge
* `notebooks/05_user_item_index_mapping.py`: train-based user/item integer mappings
* `notebooks/06_user_item_interaction_matrix.py`: sparse train/dev user-item matrix construction
* `notebooks/07_popularity_baseline.py`: static global Popularity recommender
* `notebooks/08_train_als.py`: implicit ALS training and sample recommendation
* `notebooks/09_ranking_evaluation.py`: whole-catalog ranking evaluation for Popularity and ALS
* `notebooks/10_two_stage_pipeline.py`: ALS–FAISS candidate retrieval, heuristic reranking, and two-stage evaluation on MIND-small
* `notebooks/11_parse_mind_large_streaming.py`: memory-bounded Full MIND positive-interaction extraction into partitioned Parquet files
* `notebooks/12_build_mind_large_mappings.py`: deterministic train-based Full MIND user/item mappings
* `notebooks/13_build_mind_large_sparse_matrix.py`: batched binary CSR construction for Full MIND train/dev interactions
* `notebooks/14_train_mind_large_als.py`: Full MIND implicit ALS training and factor persistence
* `notebooks/15_mind_large_recommendations.py`: Full MIND Popularity calculation and readable Popularity/ALS sample recommendations
* `notebooks/16_mind_large_ranking_evaluation.py`: Full MIND warm-start whole-catalog evaluation for Popularity and ALS
* `notebooks/17_mind_large_two_stage_pipeline.py`: Full MIND exact FAISS retrieval, top-100 candidate construction, heuristic reranking, three-model comparison, and latency evaluation

---

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Register the Jupyter kernel if using notebooks:

```bash
python -m ipykernel install --user --name recsys-news-debiasing
```

### Run ID mapping and sparse matrix construction

The current ID-mapping and matrix-construction scripts use paths relative to the `notebooks/` directory:

```bash
cd notebooks
python 05_user_item_index_mapping.py
python 06_user_item_interaction_matrix.py
cd ..
```

These scripts generate:

```text
data/processed/user_idx_map.json
data/processed/item_idx_map.json
data/processed/idx_user_map.json
data/processed/idx_item_map.json
data/processed/train_interactions.npz
data/processed/dev_interactions.npz
```

### Run the Popularity baseline

From the project root:

```bash
python notebooks/07_popularity_baseline.py
```

This generates:

```text
data/processed/popularity_scores.npy
data/processed/popularity_ranking.npy
```

### Train implicit ALS

```bash
python notebooks/08_train_als.py
```

This generates:

```text
data/processed/als_model.npz
data/processed/als_user_factors.npy
data/processed/als_item_factors.npy
```

### Evaluate Popularity and ALS

```bash
python notebooks/09_ranking_evaluation.py
```

This evaluates both models at:

```text
K = 10
K = 20
K = 40
K = 80
```

and generates:

```text
data/processed/ranking_evaluation.csv
```

---


### Run the Full MIND scale-up

Run all Full MIND scripts from the project root:

```bash
python -u notebooks/11_parse_mind_large_streaming.py
python -u notebooks/12_build_mind_large_mappings.py
python -u notebooks/13_build_mind_large_sparse_matrix.py
python -u notebooks/14_train_mind_large_als.py
python -u notebooks/15_mind_large_recommendations.py
python -u notebooks/16_mind_large_ranking_evaluation.py
python -u notebooks/17_mind_large_two_stage_pipeline.py
```

These steps perform:

```text
11: stream Full MIND behavior logs into partitioned positive interactions
12: create deterministic train-based user/item mappings
13: construct binary train/dev CSR matrices
14: train implicit ALS and save user/item factors
15: calculate Full MIND Popularity and generate sample Popularity/ALS recommendations
16: evaluate Popularity and ALS at K = 10, 20, 40, and 80
17: build the exact FAISS top-100 retrieval and heuristic reranking pipeline
```

The main Full MIND outputs are stored under:

```text
data/processed/mindlarge/
```

---

## Requirements

The current stage uses:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
jupyter
ipykernel
pyarrow
implicit
faiss-cpu
```

Later stages may add:

```text
lightfm
lightgbm
torch
sentence-transformers
fastapi
streamlit
```

---

## Current Progress

### Mathematical foundation

* [x] Project repository structure
* [x] Mathematical problem formulation
* [x] Exposure-bias formulation
* [x] Toy user-item interaction matrix
* [x] Hand-written truncated SVD
* [x] Low-rank score matrix reconstruction
* [x] Top-K recommendation generation

### MIND-small data processing

* [x] Raw data inspection
* [x] Impression parsing
* [x] User-history parsing
* [x] News metadata merge
* [x] Train-based user/item mappings
* [x] Sparse train/dev matrices
* [x] Warm-start dev filtering
* [x] Removal of train-seen dev targets

### Full MIND scale-up

* [x] Stream Full MIND train/dev behavior logs
* [x] Save positive interactions as partitioned Parquet files
* [x] Build deterministic train-based user/item mappings
* [x] Construct binary Full MIND train/dev CSR matrices
* [x] Remove cold-start and train-seen dev targets
* [x] Train 64-factor implicit ALS on Full MIND
* [x] Save the complete model and user/item factors
* [x] Generate Full MIND Popularity and ALS sample recommendations
* [x] Evaluate Full MIND Popularity and ALS at K = 10, 20, 40, and 80
* [x] Build the Full MIND FAISS `IndexFlatIP` item index
* [x] Retrieve and validate top-100 unseen candidates
* [x] Run the Full MIND heuristic reranker
* [x] Compare Popularity, ALS, and TwoStage under the same protocol
* [x] Record retrieval, reranking, and end-to-end latency
* [ ] Train a learned PyTorch reranker
* [ ] Compare MIND-small and Full MIND under the same protocol

### Classical recommendation models

* [x] Popularity baseline
* [x] Implicit ALS
* [ ] ItemKNN
* [ ] UserKNN
* [ ] Sparse truncated SVD baseline on MIND
* [ ] BPR
* [ ] LightFM

### Ranking evaluation

* [x] Recall@K
* [x] NDCG@K
* [x] MRR@K
* [x] MAP@K
* [x] Hit Rate@K
* [x] Metric sanity checks
* [x] Multiple-K evaluation
* [x] Popularity versus ALS comparison
* [x] Whole-catalog evaluation
* [ ] Impression-level evaluation
* [ ] Confidence intervals
* [ ] Statistical significance testing
* [ ] Cold-start evaluation
* [ ] Time-aware Popularity baseline

### Two-stage recommendation

* [x] Build FAISS item index
* [x] Retrieve top-100 candidates
* [x] Filter train-seen items
* [x] Verify FAISS inner-product scores
* [x] Add normalized ALS and log-popularity reranking features
* [x] Produce final top-K recommendations
* [x] Validate the ALS-only two-stage pipeline against direct ALS
* [x] Evaluate Candidate Recall@100
* [x] Evaluate final Recall, NDCG, MRR, MAP, and Hit Rate
* [x] Measure retrieval and reranking latency
* [ ] Train and evaluate a learned reranker

### Bias analysis and deployment

* [ ] Popularity bias analysis
* [ ] Exposure propensity estimation
* [ ] IPS evaluation
* [ ] SNIPS evaluation
* [ ] Doubly Robust evaluation
* [ ] FastAPI recommendation service
* [ ] Streamlit demo
* [ ] Docker packaging
* [ ] Optional Azure deployment

---

## Next Steps

Full MIND Steps 15–17 are complete:

```text
Step 15: Popularity and ALS sample recommendations
Step 16: Popularity versus ALS whole-catalog evaluation
Step 17: exact FAISS top-100 retrieval and heuristic reranking
```

The immediate next step is Step 18: train a learned PyTorch reranker instead of relying on manually selected feature weights.

The planned learned-ranking pipeline is:

```text
Full MIND train impressions
→ retain clicked and exposed-but-not-clicked candidates
→ construct candidate-level features
→ train a small PyTorch ranking model
→ apply the model to FAISS top-100 candidates
→ compare learned ranking with direct ALS and heuristic reranking
```

Initial features may include:

```text
ALS user-item dot-product score
FAISS retrieval rank
log item popularity
user history length
item training-click count
article recency
category and subcategory match
```

The ranker must be trained without using dev labels. The same warm-start dev set and the same values of $K$ will then be used for a fair comparison.

Step 19 will compare MIND-small and Full MIND in terms of:

```text
data scale
processing strategy
ALS training time
candidate retrieval latency
ranking metrics
observed temporal-popularity effects
```

Later modeling steps include:

1. implement a recent-window or time-decayed Popularity baseline;
2. tune ALS factors, regularization, alpha, and iterations;
3. add impression-level evaluation;
4. implement BPR;
5. add category and subcategory features;
6. add title and abstract embeddings;
7. evaluate cold-start news separately.

## Long-Term Project Roadmap

### Phase 1: Mathematical Foundation

* User-item interaction matrix
* Exposure bias formulation
* Low-rank latent factor model
* Toy truncated SVD recommender
* Low-rank reconstruction
* Top-K recommendation

### Phase 2: MIND Data Processing and Classical Baselines

* MIND-small data loading
* Full MIND streaming and partitioned scale-up
* User-item sparse matrix construction
* Popularity baseline
* ItemKNN / UserKNN
* Sparse truncated SVD
* ALS
* BPR
* LightFM

### Phase 3: Ranking Evaluation

* Recall@K
* Precision@K
* NDCG@K
* MRR
* MAP
* Hit Rate
* Whole-catalog evaluation
* Impression-level evaluation
* Time-based or leave-one-out evaluation
* Negative-sampling strategy
* Confidence intervals

### Phase 4: Two-Stage Recommendation System

* Candidate generation
* User and item embeddings
* FAISS-based retrieval
* Top-100 candidate generation
* Ranking features
* Ranking model
* Final top-K recommendation

### Phase 5: Deep and Content-Based Retrieval

* Two-tower retrieval
* News title embeddings
* News abstract embeddings
* Category and subcategory features
* Semantic retrieval
* Cold-start recommendation

### Phase 6: Counterfactual-Style Evaluation

* Popularity bias analysis
* Exposure propensity estimation
* IPS evaluation
* SNIPS evaluation
* Doubly Robust estimation
* Comparison between naive and propensity-weighted ranking metrics

### Phase 7: Demo and Deployment

* FastAPI recommendation endpoint
* Streamlit demo
* Docker packaging
* Optional Azure App Service deployment
* Optional Azure Blob Storage
* Optional Azure AI Search for vector retrieval

---

## Tech Stack

Current stage:

* Python
* NumPy
* pandas
* SciPy
* SciPy sparse matrices
* scikit-learn
* matplotlib
* Jupyter
* PyArrow
* implicit
* FAISS

Planned later stages:

* LightFM
* LightGBM
* PyTorch
* sentence-transformers
* Hugging Face Transformers
* FastAPI
* Streamlit
* Docker
* Azure App Service
* Azure Blob Storage
* Azure AI Search

---

## Project Goal

The goal of this project is not only to build a working recommender system, but also to explain the mathematical structure behind recommendation.

The central idea is:

$$
\text{sparse biased observations}
\rightarrow
\text{low-rank latent preference structure}
\rightarrow
\text{candidate retrieval}
\rightarrow
\text{reranking}
\rightarrow
\text{top-K recommendation}
\rightarrow
\text{debiased offline evaluation}.
$$

This project aims to connect:

```text
linear algebra
matrix factorization
implicit feedback modeling
ranking metrics
vector retrieval
counterfactual-style evaluation
modern recommendation system architecture
```

into one coherent pipeline.

The current experiments establish the first quantitative baseline:

> Under the same warm-start whole-catalog evaluation protocol, implicit ALS consistently outperformed the static global Popularity baseline at K = 10, 20, 40, and 80. However, the low absolute metrics show that retrieving future clicked news from a rapidly changing catalog of more than 51,000 items remains difficult.

The Full MIND extension reaches the same qualitative conclusion over 711,222 users and 101,527 items. Exact FAISS `IndexFlatIP` retrieval reproduces ALS scores and rankings, while adding a 1% static log-popularity contribution slightly reduces ranking quality. This result motivates time-aware features and a learned reranker rather than stronger reliance on cumulative historical popularity.
