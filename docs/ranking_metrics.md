## Ranking Metrics

Let:

* $G_u$ be the set of relevant items for user $u$.
* $R_u^K$ be the set of items in the top-$K$ recommendation list.
* $\pi_u(r)$ be the item ranked at position $r$.
* $rel_u(r)=1$ if $\pi_u(r)\in G_u$, and $rel_u(r)=0$ otherwise.
* $N$ be the number of evaluated users.

We use the following example throughout this section:

```text
Relevant items: {A, B, C}

Top-5 recommendations:
Rank 1: X
Rank 2: A
Rank 3: Y
Rank 4: B
Rank 5: Z
```

The model retrieves two relevant items, $A$ and $B$, at ranks 2 and 4.

---

### Recall@K — Recall at K

For user $u$:

$Recall@K(u)=\frac{|G_u\cap R_u^K|}{|G_u|}$

The overall Recall@K is the average across evaluated users:

$Recall@K=\frac{1}{N}\sum_{u=1}^{N}Recall@K(u)$

**Definition:** Recall@K is the fraction of a user’s relevant items that appear in the top-$K$ recommendation list.

**What it measures:** It measures how many relevant items the model successfully retrieves.

**Example:**

The user has three relevant items:

$G_u={A,B,C}$

The top-5 list retrieves two of them:

$R_u^5\cap G_u={A,B}$

Therefore:

$Recall@5=\frac{2}{3}\approx0.667$

Recall does not consider the exact ranking positions. It only checks whether relevant items appear somewhere in the top-$K$ list.

For example, the following two lists have the same Recall@5:

```text
[A, B, X, Y, Z]
[X, Y, Z, A, B]
```

Both lists retrieve two of the three relevant items, even though the first ranking is clearly better.

---

### NDCG@K — Normalized Discounted Cumulative Gain at K

First, calculate Discounted Cumulative Gain:

$DCG@K(u)=\sum_{r=1}^{K}\frac{rel_u(r)}{\log_2(r+1)}$

A relevant item receives a smaller contribution when it appears at a lower rank.

The ideal DCG is:

$IDCG@K(u)=\sum_{r=1}^{\min(|G_u|,K)}\frac{1}{\log_2(r+1)}$

Then:

$NDCG@K(u)=\frac{DCG@K(u)}{IDCG@K(u)}$

The overall NDCG@K is:

$NDCG@K=\frac{1}{N}\sum_{u=1}^{N}NDCG@K(u)$

**Definition:** NDCG@K is DCG@K normalized by the best possible DCG@K for the same user.

**What it measures:** It measures both whether relevant items are retrieved and whether they are ranked near the top.

**Example:**

The relevant items appear at ranks 2 and 4.

Therefore:

$DCG@5=\frac{1}{\log_2(3)}+\frac{1}{\log_2(5)}$

$DCG@5\approx0.631+0.431=1.062$

The user has three relevant items, so the ideal ranking would place them at ranks 1, 2, and 3:

$IDCG@5=1+\frac{1}{\log_2(3)}+\frac{1}{\log_2(4)}$

$IDCG@5\approx1+0.631+0.500=2.131$

Therefore:

$NDCG@5=\frac{1.062}{2.131}\approx0.498$

NDCG is usually between 0 and 1:

```text
NDCG = 1:
All relevant items are ranked in the best possible positions.

NDCG = 0:
No relevant item appears in the top-K list.
```

Unlike Recall, NDCG distinguishes between a relevant item at rank 1 and the same item at rank 10.

---

### MRR@K — Mean Reciprocal Rank at K

For one user, let $r_u$ be the rank of the first relevant item.

The reciprocal rank is:

$RR@K(u)=\frac{1}{r_u}$

if the first relevant item appears within the top-$K$ list.

If no relevant item appears in the top-$K$ list:

$RR@K(u)=0$

The mean reciprocal rank is:

$MRR@K=\frac{1}{N}\sum_{u=1}^{N}RR@K(u)$

**Definition:** MRR@K is the average reciprocal rank of the first relevant item across users.

**What it measures:** It measures how early the first relevant recommendation appears.

**Example:**

The first relevant item, $A$, appears at rank 2.

Therefore:

$RR@5=\frac{1}{2}=0.5$

Examples of reciprocal rank:

```text
First relevant item at rank 1: RR = 1
First relevant item at rank 2: RR = 1/2
First relevant item at rank 5: RR = 1/5
No relevant item in top-K:     RR = 0
```

MRR only considers the first relevant item.

For example:

```text
[A, X, Y, Z, W]
[A, B, C, X, Y]
```

Both lists have:

$RR@5=1$

because the first relevant item appears at rank 1 in both lists. MRR does not reward the second list for retrieving more relevant items.

---

### MAP@K — Mean Average Precision at K

First, define Precision at rank $r$:

$Precision@r(u)=\frac{\sum_{j=1}^{r}rel_u(j)}{r}$

Average Precision at K is:

$AP@K(u)=\frac{\sum_{r=1}^{K}Precision@r(u)\cdot rel_u(r)}{\min(|G_u|,K)}$

Mean Average Precision is:

$MAP@K=\frac{1}{N}\sum_{u=1}^{N}AP@K(u)$

**Definition:** MAP@K is the average of user-level Average Precision values.

**What it measures:** It measures how consistently multiple relevant items are ranked near the top of the recommendation list.

**Example:**

The relevant items appear at ranks 2 and 4.

At rank 2:

```text
One of the first two recommendations is relevant.
```

Therefore:

$Precision@2=\frac{1}{2}=0.5$

At rank 4:

```text
Two of the first four recommendations are relevant.
```

Therefore:

$Precision@4=\frac{2}{4}=0.5$

The user has three relevant items, so:

$AP@5=\frac{Precision@2+Precision@4}{3}$

$AP@5=\frac{0.5+0.5}{3}\approx0.333$

MAP considers all relevant items that appear in the recommendation list, unlike MRR, which only considers the first one.

Relevant items appearing earlier produce higher precision values and therefore a higher MAP score.

---

### Hit Rate@K — Hit Rate at K

For one user:

$Hit@K(u)=1$

if at least one relevant item appears in the top-$K$ recommendation list.

Otherwise:

$Hit@K(u)=0$

The overall Hit Rate is:

$HitRate@K=\frac{1}{N}\sum_{u=1}^{N}Hit@K(u)$

Equivalently:

$HitRate@K=\frac{\text{number of users with at least one hit}}{\text{number of evaluated users}}$

**Definition:** Hit Rate@K is the fraction of users whose top-$K$ recommendation list contains at least one relevant item.

**What it measures:** It measures whether the model produces at least one successful recommendation for each user.

**Example:**

The top-5 recommendations contain two relevant items, $A$ and $B$.

Therefore:

$Hit@5=1$

Hit Rate does not distinguish between one hit and multiple hits.

For example:

```text
[X, A, Y, Z, W]
[A, B, C, X, Y]
```

Both lists have:

$Hit@5=1$

because both contain at least one relevant item.

If 100 out of 1,000 evaluated users receive at least one relevant item in their top-10 recommendations:

$HitRate@10=\frac{100}{1000}=0.1$

---

## Summary

| Metric     | Full Name                                  | Main Question                                                  | Considers Ranking Position? | Considers Multiple Relevant Items? |
| ---------- | ------------------------------------------ | -------------------------------------------------------------- | --------------------------: | ---------------------------------: |
| Recall@K   | Recall at K                                | How many relevant items were retrieved?                        |                          No |                                Yes |
| NDCG@K     | Normalized Discounted Cumulative Gain at K | Were relevant items retrieved and ranked near the top?         |                         Yes |                                Yes |
| MRR@K      | Mean Reciprocal Rank at K                  | How early did the first relevant item appear?                  |                         Yes |                                 No |
| MAP@K      | Mean Average Precision at K                | Were multiple relevant items consistently ranked near the top? |                         Yes |                                Yes |
| Hit Rate@K | Hit Rate at K                              | Did the model retrieve at least one relevant item?             |                          No |                                 No |

A simple way to remember the metrics is:

```text
Recall@K:
How many relevant items were found?

NDCG@K:
How many were found, and how highly were they ranked?

MRR@K:
How early was the first hit?

MAP@K:
How well were all hits ranked?

Hit Rate@K:
Was there at least one hit?
```
