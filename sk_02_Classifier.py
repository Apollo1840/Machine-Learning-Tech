# -*- coding: utf-8 -*-

# load data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target)


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(C=100)  # C is coefficient for cost, smaller means it care more about the majority, default is 1
# the previous model seems to be under fitting
logr.fit(x_train, y_train)
print(logr.score(x_train, y_train))
print(logr.score(x_test, y_test))

# accuracy
from sklearn.metrics import accuracy_score
y_pred = logr.predict(x_train)
print(accuracy_score(y_train, y_pred))

from sklearn.metrics import precision
print(precision(y_train, y_pred))

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_train, y_pred, target_names=target_names))




print('coeffcients:')
print(logr.coef_)
print(logr.intercept_)
print('\n')

print(logr.decision_function(x_test[:20]))
print(logr.predict_proba(x_test[:20]))
print(logr.predict(x_test[:20]))


# plot the classification for 2D classifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target


model = LogisticRegression(C=1e5)
model.fit(X, Y)
 

# Put the result into a color plot
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

# multiple assignment
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# now color it:
h = 0.02  # step size in the mesh

# try: np.meshgrid([1,2,3,4],[1,2,3,4])
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# try: np.c_[[1,1,1,1],[2,2,2,2]]
X_mesh = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X_mesh)
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.show()



# other classifiers

#############################################################
from sklearn.svm import SVC

svm=SVC()
svm.fit(x_train, y_train)
print(svm.score(x_train, y_train))
print(svm.score(x_test, y_test))

# adjust the model:
# Kernel
svm = SVC(kernel='rbf', gamma=1) # default one, the default gamma is 1/n_features

svm = SVC(kernel='linear')  # most simple one
svm = SVC(kernel='poly', degree=2)  # polynomial kernel usually used in NLP
svm = SVC(kernel='sigmoid') 

svm = SVC(kernel= distance_matrix)
 
# strategy
svm = SVC(shrinking=False)  

svm = SVC(probablity = True)  # this is necessary for svc to predict_proba, it will slow down the train and predict 



#############################################################
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(5)
knn.fit(x_train, y_train)
print(knn.score(x_train, y_train))
print(knn.score(x_test, y_test))


#############################################################
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(x_train, y_train)
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))

print(tree.feature_importances_)

#############################################################
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(100, n_jobs=-1)
forest.fit(x_train, y_train)

print(forest.score(x_train, y_train))
print(forest.score(x_test, y_test))


#############################################################
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(0.01, probability=True)
gbc.fit(x_train, y_train)
gbc.predict_proba(x_test[:20])

#############################################################
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
mlp.fit(x_train, y_train)
print(mlp.score(x_train, y_train))
print(mlp.score(x_test, y_test))

#############################################################
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
y_pred_p = classifier.fit(x_train, y_train).predict_proba(x_test)
print(y_pred_p)


'''
    1, you need to know the following classifier and where it from:
        
        sklearn.linear_model      --    LogisticRegression
        sklearn.svm               --    SVC
        sklearn.neighbors         --    KNeighborsClassifier
        
        sklearn.tree              --    DecisionTreeClassifier
        sklearn.ensemble          --    RandomForestClassifier        
        sklearn.ensemble          --    GradientBoostingClassifier
  
        sklearn.neural_network    --    MLPClassifier
        sklearn.multiclass        --    OneVsRestClassifier
        
    2, you need to those attribute of a classifier:
        
        .coef_
        .intercept_
        .predict
        .predict_proba
        .decision_function
    
    3, you need to know how to draw the 2D classifier
    
    4, details
        1) For logisticRegression:
            C
            
        2) For SVC
            C
            kernel
            probability
            
        3) For KNeighborsClassifier
    
    5, you need to know how to use OneVsRestClassifier
    
    6, how to evaluate the model, from metrics (see more in http://scikit-learn.org/stable/modules/model_evaluation.html)
        accuracy_score()
        precision()  
        
            
        
'''