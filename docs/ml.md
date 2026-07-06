## 1. ML 基础概念类

### 1.1 General ML Concepts

* [ ] 什么是 overfitting？
* [ ] 什么是 underfitting？
* [ ] 什么是 bias-variance trade-off？
* [ ] 过拟合一般有哪些预防手段？
* [ ] Generative model 和 discriminative model 的区别是什么？
* [ ] 给定一组 ground truth 和两个模型，如何判断一个模型显著优于另一个模型？
* [ ] 如果训练好的模型在现实中不 work，可能有哪些原因？
* [ ] 如果 production data 和 development data 发生 shift，如何 detect 和补救？

---

## 2. Regularization

* [ ] L1 regularization 和 L2 regularization 分别是什么？
* [ ] L1 和 L2 的区别是什么？
* [ ] Lasso 和 Ridge 的解释是什么？
* [ ] Lasso 和 Ridge 分别对应什么 prior？
* [ ] Lasso / Ridge 如何推导？
* [ ] 为什么 L1 比 L2 更容易产生稀疏解？
* [ ] 为什么 regularization 有效？
* [ ] 为什么 regularization 常用 L1 / L2，而不是 L3 / L4？

---

## 3. Metrics and Evaluation

### 3.1 Classification Metrics

* [ ] Precision 是什么？
* [ ] Recall 是什么？
* [ ] Precision 和 recall 的 trade-off 是什么？
* [ ] Label 不平衡时应该用什么 metric？
* [ ] 分类问题应该选什么 metric？为什么？
* [ ] Confusion matrix 是什么？
* [ ] True positive rate 是什么？
* [ ] False positive rate 是什么？
* [ ] ROC curve 是什么？
* [ ] AUC 如何解释？
* [ ] AUC 为什么可以理解为：随机选一个正样本和一个负样本，模型把正样本排在负样本前面的概率？
* [ ] Log-loss 是什么？
* [ ] 什么时候用 log-loss？

### 3.2 Ranking / Recommendation Metrics

* [ ] Ranking 任务应该用什么 metric？
* [ ] 推荐系统应该用什么 metric？
* [ ] 如何根据具体业务场景选择 metric？

---

## 4. Loss Functions and Optimization

* [ ] MSE 的公式是什么？
* [ ] MSE 什么时候使用？
* [ ] 用 MSE 作为 logistic regression 的 loss，是 convex problem 吗？
* [ ] Linear regression 中，最小二乘法和 maximum likelihood estimation 的关系是什么？
* [ ] Relative entropy 是什么？
* [ ] Cross entropy 是什么？
* [ ] KL divergence 是什么？
* [ ] Cross entropy 和 KL divergence 的 intuition 是什么？
* [ ] Logistic regression 的 loss 是什么？
* [ ] Logistic regression 的 loss 如何推导？
* [ ] SVM 的 loss 是什么？
* [ ] Multiclass logistic regression 为什么用 cross entropy 作为 cost function？
* [ ] Decision tree split node 时优化目标是什么？

---

## 5. Deep Learning 基础概念类

### 5.1 Neural Network Basics

* [ ] DNN 为什么需要 bias term？
* [ ] Bias term 的 intuition 是什么？
* [ ] 什么是 backpropagation？
* [ ] DNN 和 logistic regression 的区别是什么？
* [ ] 为什么 DNN 的拟合能力比 logistic regression 强？
* [ ] 为什么 neural network 需要 non-linear activation function？
* [ ] Neural network 的 weights 能不能全部 initialize 成 0？为什么？
* [ ] Transfer learning 什么时候有意义？

### 5.2 Gradient Problems

* [ ] 什么是 gradient vanishing？
* [ ] 什么是 gradient exploding？
* [ ] Gradient vanishing / exploding 如何解决？
* [ ] Plateau 是什么问题？
* [ ] Saddle point 是什么问题？

### 5.3 Activation Functions

* [ ] Sigmoid 是什么？优缺点是什么？
* [ ] Tanh 是什么？优缺点是什么？
* [ ] ReLU 是什么？优缺点是什么？
* [ ] Leaky ReLU 是什么？优缺点是什么？
* [ ] 不同 activation function 如何选择？

### 5.4 Optimization in Deep Learning

* [ ] SGD 是什么？
* [ ] Batch gradient descent 是什么？
* [ ] Mini-batch gradient descent 是什么？
* [ ] Batch 和 SGD 的优缺点是什么？
* [ ] Batch size 对训练有什么影响？
* [ ] Momentum 是什么？
* [ ] RMSprop 是什么？
* [ ] Adagrad 是什么？
* [ ] Adam 是什么？
* [ ] 不同 optimizer 的区别是什么？
* [ ] Learning rate 过大有什么影响？
* [ ] Learning rate 过小有什么影响？

### 5.5 Overfitting in Deep Learning

* [ ] Deep learning 中有哪些防止 overfitting 的办法？
* [ ] Dropout 是什么？
* [ ] Dropout 为什么有效？
* [ ] Dropout 的训练流程是什么？
* [ ] Dropout 在 training 和 testing 时有什么区别？
* [ ] Batch normalization 是什么？
* [ ] Batch normalization 为什么有效？
* [ ] Batch normalization 的流程是什么？
* [ ] Batch normalization 在 training 和 testing 时有什么区别？
* [ ] 如何做 hyperparameter tuning？
* [ ] Random search 和 grid search 的区别是什么？

---

## 6. ML 模型类

### 6.1 Linear Regression

* [ ] Linear regression 的基础假设是什么？
* [ ] Regression coefficient 如何解释？
* [ ] Minimizing squared error 和 maximizing likelihood 的关系是什么？
* [ ] 如果变量之间高度相关，会发生什么？
* [ ] 如何解决 correlated variables 的问题？
* [ ] 如何 minimize inter-correlation between variables with linear regression？
* [ ] 如果 y 和 x 的关系不是线性的，linear regression 能解决吗？
* [ ] 为什么使用 interaction variables？

### 6.2 Logistic Regression

* [ ] Logistic regression 和 SVM 的区别是什么？
* [ ] Logistic regression 输出的是 probability，SVM 输出的是 score，这个区别意味着什么？
* [ ] Logistic regression 的 log-loss 是什么？
* [ ] Logistic regression 中 regularization 如何使用？

### 6.3 SVM and Kernel Methods

* [ ] Explain SVM.
* [ ] SVM 如何引入非线性？
* [ ] Kernel method 是什么？
* [ ] 为什么使用 kernel？
* [ ] 常见 kernel 有哪些？
* [ ] 怎么把 SVM 的 output 转成概率输出？

### 6.4 KNN

* [ ] Explain KNN.
* [ ] KNN 的优缺点是什么？
* [ ] KNN 对 feature scaling 是否敏感？
* [ ] KNN 在高维数据中有什么问题？

### 6.5 PCA

* [ ] Explain PCA.
* [ ] PCA 的目标是什么？
* [ ] PCA 的优缺点是什么？

### 6.6 K-means / Clustering / EM

* [ ] Explain K-means algorithm in detail.
* [ ] K-means 是否会收敛？
* [ ] K-means 收敛到 global optimum 还是 local optimum？
* [ ] K-means 如何停止？
* [ ] EM algorithm 是什么？
* [ ] GMM 是什么？
* [ ] GMM 和 K-means 的关系是什么？

### 6.7 Decision Tree

* [ ] Classification decision tree 如何 split nodes？
* [ ] Regression decision tree 如何 split nodes？
* [ ] Decision tree 如何防止 overfitting？
* [ ] Decision tree 如何 regularize？
* [ ] Decision tree 的优缺点是什么？

### 6.8 Ensemble Learning

* [ ] Bagging 和 boosting 的区别是什么？
* [ ] Random forest 是什么？
* [ ] GBDT 是什么？
* [ ] Random forest 和 GBDT 的区别是什么？
* [ ] Random forest 和 GBDT 的 pros and cons 是什么？
* [ ] Random forest 是降低 bias 还是 variance？
* [ ] 为什么 random forest 可以降低 variance？
* [ ] Explain GBDT.
* [ ] Explain random forest.

### 6.9 Generative Models

* [ ] Generative model 和 discriminative model 相比，更容易 overfitting 还是 underfitting？
* [ ] Naive Bayes 的原理是什么？
* [ ] Naive Bayes 的基础假设是什么？
* [ ] LDA 是什么？
* [ ] QDA 是什么？
* [ ] LDA / QDA 的假设是什么？

### 6.10 Model Pros and Cons

* [ ] 所有简历里提到的模型都要能讲 pros and cons。
* [ ] 所有 JD 里提到的模型都要能讲 pros and cons。
* [ ] 面试组 domain 相关的常用模型要能讲 pros and cons。

---

## 7. 数据处理类

* [ ] 如何处理 imbalanced data？
* [ ] High-dimensional classification 有什么问题？
* [ ] High-dimensional classification 如何处理？
* [ ] Missing data 如何处理？
* [ ] 如何做 feature selection？
* [ ] 如何 capture feature interaction？
* [ ] Annotation 有限的情况下如何 train model？
* [ ] 如果 production 中一个 important feature missing，且不能重新 train model，怎么办？

---

## 8. Implementation / Coding / Derivation 类

* [ ] 写代码实现两层 fully connected neural network。
* [ ] 手写 CNN。
* [ ] 手写 KNN。
* [ ] 手写 K-means。
* [ ] 手写 softmax 的 backpropagation。
* [ ] 给定 LSTM network 结构，计算有多少 parameters。
* [ ] Convolution layer 的 output size 怎么算？
* [ ] 写出 convolution layer output size 的公式。
* [ ] 不用 package 手写一个简单的 GCN，如果简历里写了 Graph Convolutional Neural Network。

---

## 9. Production / Project Experience 类

* [ ] 训练好的模型在现实中不 work，可能原因是什么？
* [ ] Loss 趋于 Inf 的可能原因是什么？
* [ ] Loss 出现 NaN 的可能原因是什么？
* [ ] Production 和 development data 发生 shift，如何 detect？
* [ ] Production 和 development data 发生 shift，如何补救？
* [ ] Annotation 有限时如何 train model？
* [ ] Model 要放 production，但 online 一个 important feature missing，并且不能重新 train model，怎么办？

---

## 10. NLP / RNN 类

### 10.1 RNN / LSTM

* [ ] 为什么使用 RNN？
* [ ] 为什么使用 LSTM？
* [ ] LSTM 的公式是什么？
* [ ] LSTM 比 vanilla RNN 好在哪里？
* [ ] RNN 的 limitation 是什么？
* [ ] RNN 中 gradient vanishing 如何解决？

### 10.2 Attention

* [ ] Attention 是什么？
* [ ] 为什么使用 attention？
* [ ] Traditional attention mechanism 是什么？
* [ ] Self-attention 和 traditional attention 的区别是什么？

### 10.3 Language Model

* [ ] Language model 的原理是什么？
* [ ] N-gram model 是什么？
* [ ] N-gram model 的 limitation 是什么？

### 10.4 Word Embedding

* [ ] Word2Vec 是什么？
* [ ] CBOW 是什么？
* [ ] Skip-gram 是什么？
* [ ] Word2Vec 的 loss function 是什么？
* [ ] Negative sampling 是什么？
* [ ] 为什么需要 negative sampling？

### 10.5 Transformer / BERT / NLP Domain Questions

* [ ] What is Transformer?
* [ ] Explain Transformer architecture.
* [ ] What is BERT?
* [ ] Explain BERT architecture.
* [ ] Transformer / BERT 比 LSTM 好在哪里？
* [ ] Self-attention 和 traditional attention mechanism 的区别是什么？
* [ ] 简单的 model distillation 方法有哪些？
* [ ] 如果面 ASR 方向，需要了解哪些 widely used models / methods？
* [ ] 如果面 chatbot 方向，需要了解哪些 widely used models / methods？

---

## 11. CNN / CV 类

### 11.1 CNN Basics

* [ ] Convolution layer 是什么？
* [ ] 为什么使用 convolution layer？
* [ ] Max pooling 是什么？
* [ ] 为什么做 pooling？
* [ ] CNN 为什么适合图像任务？
* [ ] 什么是 equivariant to translation？
* [ ] 什么是 invariant to translation？
* [ ] 1x1 filter 是什么？
* [ ] 1x1 convolution 有什么作用？
* [ ] Skip connection 是什么？

### 11.2 CV Domain Questions

* [ ] 如果面 CV segmentation 组，需要了解 U-Net。
* [ ] 如果面 CV segmentation 组，可能需要了解 DeepMask。
* [ ] CV 方向需要熟悉本组 domain 里 widely used 的模型和方法。

