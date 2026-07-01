# Day 1：低秩推荐系统的问题定义与最小可运行 SVD 示例

## 今日定位

建立项目骨架，写清楚数学问题定义，并跑通一个最小的低秩推荐 toy example。

这个项目的长期主线是：

> 稀疏且带偏的 user-item interaction matrix  
> → 低秩矩阵分解  
> → top-K ranking evaluation  
> → exposure bias / popularity bias analysis  
> → two-stage retrieval + ranking  
> → demo / API / optional Azure deployment

Day 1 只做第一段：

> user-item matrix → low-rank structure → SVD toy recommender → README 数学叙事

---

## 今日最终交付物

今天结束时，项目 repo 至少应该有以下文件：

```text
recsys-news-debiasing/
├── README.md
├── requirements.txt
├── docs/
│   └── 01_problem_formulation.md
├── notebooks/
│   └── 01_low_rank_recommender_toy_example.ipynb
└── src/
    ├── data/
    ├── models/
    ├── evaluation/
    ├── retrieval/
    └── ranking/
```

今天的最低完成标准：

1. 项目目录创建完成。
2. `requirements.txt` 写好并能安装。
3. `docs/01_problem_formulation.md` 写出第一版数学定义。
4. `notebooks/01_low_rank_recommender_toy_example.ipynb` 跑通 toy SVD 推荐。
5. `README.md` 有清楚的项目定位和当前进度。

---

## 任务 1：创建项目结构

在本地运行：

```bash
cd /Users/hl/Desktop/PythonProject1/mind

mkdir -p docs notebooks src/data src/models src/evaluation src/retrieval src/ranking

touch README.md
touch requirements.txt
touch docs/01_problem_formulation.md
```

建议今天先不要创建太多复杂文件。目录可以先有，代码文件后面逐步加。

---

## 任务 2：写 requirements.txt

今天只安装数学 baseline 需要的依赖：

```txt
numpy
pandas
scipy
scikit-learn
matplotlib
jupyter
ipykernel
```

安装环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name recsys-news-debiasing
```

今日不需要安装：

- `implicit`
- `lightfm`
- `faiss`
- `torch`
- `sentence-transformers`
- `fastapi`
- `streamlit`
- Azure 相关 SDK

这些都留到后面的阶段。

---

## 任务 3：写 docs/01_problem_formulation.md

这个文档是项目的数学地基。今天不需要写得像论文，但必须把核心概念讲清楚。

建议结构如下：

```md
# Problem Formulation

## 1. Recommendation as Matrix Completion

Let there be \(m\) users and \(n\) news items. Define the user-item interaction matrix

\[
R \in \{0,1\}^{m \times n}.
\]

Here, \(R_{ui}=1\) means user \(u\) clicked item \(i\). However, in implicit feedback recommendation, \(R_{ui}=0\) does not necessarily mean the user dislikes the item. It may simply mean the user was never exposed to it.

## 2. Exposure Bias

A more accurate observation model is

\[
R_{ui} = E_{ui}Y_{ui},
\]

where:

- \(E_{ui}=1\) means item \(i\) was exposed to user \(u\);
- \(Y_{ui}=1\) means user \(u\) would click or like item \(i\);
- \(R_{ui}\) is the observed click.

In matrix form:

\[
R = E \odot Y.
\]

The recommender system wants to learn the hidden preference matrix \(Y\), but only observes the biased click matrix \(R\).

## 3. Low-Rank Latent Factor Model

Assume each user and item can be represented in a shared \(k\)-dimensional latent space:

\[
p_u \in \mathbb{R}^k, \quad q_i \in \mathbb{R}^k.
\]

The predicted preference score is

\[
s(u,i) = p_u^\top q_i.
\]

In matrix form:

\[
S = P Q^\top,
\]

where \(P\) is the user embedding matrix and \(Q\) is the item embedding matrix. Since \(S = P Q^\top\), we have

\[
\operatorname{rank}(S) \leq k.
\]

This is the low-rank structure behind collaborative filtering.

## 4. Implicit Matrix Factorization Objective

For implicit feedback, one common objective is

\[
\min_{P,Q}
\sum_{u,i} c_{ui}(R_{ui} - p_u^\top q_i)^2
+
\lambda(\|P\|_F^2 + \|Q\|_F^2),
\]

where \(c_{ui}\) is a confidence weight. A common choice is

\[
c_{ui} = 1 + \alpha R_{ui}.
\]

Clicked interactions receive higher confidence, while unclicked interactions are treated as low-confidence observations rather than strong negatives.
```

今日重点：

- 一定要写出 \(R = E \odot Y\)。
- 一定要写出 \(S = P Q^\top\)。
- 一定要解释为什么 unclicked 不等于 dislike。
- 一定要解释为什么这是低秩问题。

---

## 任务 4：创建 toy SVD notebook

创建：

```text
notebooks/01_low_rank_recommender_toy_example.ipynb
```

Notebook 的目标不是追求性能，而是展示推荐系统的数学直觉。

### 4.1 构造 toy user-item matrix

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

users = ["u1", "u2", "u3", "u4", "u5"]
items = ["Politics", "Tech", "Sports", "Finance", "Health", "Travel"]

R = np.array([
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 0],
])

df = pd.DataFrame(R, index=users, columns=items)
df
```

### 4.2 做 truncated SVD

```python
k = 2

svd = TruncatedSVD(n_components=k, random_state=42)
P = svd.fit_transform(R)
Q = svd.components_.T

S_hat = P @ Q.T
score_df = pd.DataFrame(S_hat, index=users, columns=items)
score_df
```

### 4.3 生成 top-K 推荐

```python
def recommend_for_user(user_id, R, scores, users, items, top_k=3):
    u_idx = users.index(user_id)
    seen = R[u_idx] > 0

    user_scores = scores[u_idx].copy()
    user_scores[seen] = -np.inf

    top_indices = np.argsort(user_scores)[::-1][:top_k]
    return [(items[i], float(user_scores[i])) for i in top_indices]

recommend_for_user("u1", R, S_hat, users, items, top_k=3)
```

### 4.4 Notebook 里要写的解释

在 notebook markdown cell 里写：

```md
The original user-item matrix is sparse. However, users and news items may share a low-dimensional latent structure. For example, users who click Politics and Tech may be close in latent space, while users who click Sports and Health may form another group.

Truncated SVD approximates the sparse interaction matrix with a low-rank score matrix. The reconstructed scores can be used to recommend unseen items with high predicted preference.
```

中文理解：

> 原始点击矩阵很稀疏，但用户和新闻之间可能存在低维兴趣结构。SVD 通过低秩逼近恢复这种 latent structure，从而可以给用户推荐尚未点击但可能感兴趣的新闻类别。

---

## 任务 5：写 README.md 第一版

README 今天只需要写项目定位，不需要写完整实验结果。

建议内容：

```md
# Debiased News Recommendation via Low-Rank Matrix Factorization and Counterfactual Evaluation

This project builds a mathematically grounded news recommendation system based on low-rank matrix factorization, ranking-based evaluation, and counterfactual-style debiasing.

The project treats recommendation not simply as click prediction, but as learning latent user-item preference structure from sparse and biased implicit feedback data.

## Core Ideas

- User-item interaction matrix
- Low-rank matrix factorization
- Implicit feedback modeling
- Exposure bias and popularity bias
- Ranking metrics such as Recall@K, NDCG@K, MRR, and MAP
- Counterfactual-style offline evaluation using IPS and SNIPS
- Two-stage retrieval and ranking architecture

## Mathematical View

Given a user-item interaction matrix

\[
R \in \{0,1\}^{m \times n},
\]

we model user preferences using a low-rank score matrix

\[
S = P Q^\top.
\]

Observed clicks are biased because users only interact with exposed items:

\[
R = E \odot Y.
\]

## Current Progress

- [x] Project formulation
- [x] Toy low-rank recommender
- [ ] MIND-small data processing
- [ ] Popularity baseline
- [ ] ItemKNN / UserKNN
- [ ] Truncated SVD baseline
- [ ] ALS
- [ ] BPR
- [ ] LightFM
- [ ] Ranking evaluation
- [ ] Counterfactual evaluation
- [ ] Two-stage retrieval and ranking
- [ ] FastAPI / Streamlit demo
```

---

## 今日不要做的事情

为了避免项目一开始就失控，今天明确不要做：

1. 不下载 MIND-full。
2. 不做 Azure。
3. 不写 FastAPI。
4. 不写 Streamlit。
5. 不做 two-tower。
6. 不做 LLM embedding。
7. 不做复杂的 debiased evaluation。
8. 不纠结模型指标。
9. 不引入太多依赖。
10. 不把 README 写成空泛项目介绍。

今天只做数学地基和最小可运行 demo。

---

## 今日完成后的自检问题

完成 Day 1 后，你应该能回答这些问题：

1. 为什么推荐系统可以看成 user-item matrix 问题？
2. 为什么 \(R_{ui}=0\) 不一定表示用户不喜欢 item？
3. \(R = E \odot Y\) 里的 \(E\) 和 \(Y\) 分别是什么意思？
4. 为什么矩阵分解对应低秩结构？
5. \(S = P Q^\top\) 里的 \(P\) 和 \(Q\) 分别是什么？
6. SVD toy example 是如何生成推荐的？
7. 为什么今天不直接做 deep learning？

如果这些问题都能讲清楚，Day 1 就成功了。

---

## 今日结束时的 Git commit 建议

如果你使用 Git，今天结束时可以提交：

```bash
git init
git add .
git commit -m "Day 1: initialize low-rank recommender foundation"
```

---

## Day 1 成功标准

今天不是要证明模型效果，而是要证明项目方向成立。

Day 1 成功的标志是：

> 这个项目已经有了清楚的数学问题定义、可运行的低秩推荐 toy example、以及一个能继续扩展到真实 MIND 数据的 repo 骨架。

下一步 Day 2 才进入：

> MIND-small 数据下载、`behaviors.tsv` / `news.tsv` 解析、user-item interaction matrix 构建。
