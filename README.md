# Machine-Learning-Tech
There are examples of how to use Scikit learn to do Machine Learnings and how to use tensorflow and keras to do deep learning.


# 0. Machine Learning Handbook
Basically, there are three categories of machine learning.

* Supervised learning
* Unsupervised learning
* Reinforcement learning

Under supervise learning, there are regression, classification ...

Under Unsuperives learning, there are clustering, GAN(Generative Adversarial Network) ...


Dont forget, for ML algorithm, there are some question should be thought in advance:

* quality of the data?
* how to choose training data, validation data and test data?
* which attribute to choose?
* how to evaluate the model?
* in which situation the algorithm is going to be used, what should be take care of?

## 0.0 Advance

### 0.0.0 data quality
outlier

### 0.0.1 factor selection

Why should we reduce the number of the attributes? 
It is like Why should we use l1 regularization.

AIC or BIC:

At large n, AIC tends to pick somewhat larger models than BIC. 
If you're trying to understand what the main drivers are, you might want something more like BIC. 
If that's less important than good MSPE, you might lean more toward AIC.


### 0.0.2 train/dev/test
60/20/20 or 98/1/1

leverage point

k-fold cross validation

### 0.0.3 evaluation

| TP | FP | TN | FN |
| ---- | ---- | ---- | ---- | 

accuracy: T/A

sensitivity: TP/P

specificity: TN/N

ROC: sensitivity(x) to 1-specificity(y)

Imagine, you have several apples, some are good, some are bad. You ask your child (the model) to pick them: keep the good apples, and throw the bad apples. 

True is the good apples kept and the bad apples thrown.

False is the bad apples kept (Error I) and the good apples thrown (Error II).

Positive is the apples your child keep.

Negative is the apples your child thrown.

Accuracy is the True/All

There are two aspect to evaluate the model: precision and recall.

Precision is easy.

postive precision is the ratio of good apples compare to what your chlid kept.

negative precision is the ratio of of bad apples compare to what your child thrown.

Recall is more interesting.

positve recall (sensitivity) is the ratio of good apples you have compare to all good apples. 

negative recall (specifictiy) is the of ratio of bad apples your child thrown compare to all bad apples. 




## 0.1 Supervise learning

### 0.1.1 Regression
Linear regression

Polynomial regression

Advance:

Linear regression with AIC step

Ridge regression; Lasson regression

Robust regression

GradientBoosting

Xgboosting

### 0.0.2 Classification
Decision tree (forest)

Logistic Regression

SVM

SVM with kernel

kNN

NN (MLPClassifier: Multi-layer Perceptron)

AdaBoosting

## 0.2 Unsupervise learning

### 0.2.1 clustering
K-mean

Hierarchy

DBSCAN [https://www.jianshu.com/p/e8dd62bec026]

GMM [https://blog.csdn.net/jwh_bupt/article/details/7663885]

FCM

SOM


### 0.2.2 GAN

## 0.3 Reinforcement learning


# 1. Deep learning handbook


## 1.0 basic improvements

## 1.0.1 low biase

## 1.0.2 low variance

regularization

dropout

Input Normalization

Batch Normalization

Parameter tunning



## Do you know?

what is GradientBoostingClassifier?

what is shrinking mean in SVM?


 