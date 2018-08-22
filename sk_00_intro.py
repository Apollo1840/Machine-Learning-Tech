
# show ML
import mglearn
mglearn.plots.plot_knn_classification(n_neighbors=3)
mglearn.plots.plot_linear_regression_wave()
mglearn.plots.plot_tree_not_monotone()
mglearn.plots.plot_two_hidden_layer_graph()
###############################################################################



# preparation 1: matplotlib
import matplotlib
% matplotlib inline

import matplotlib.pyplot as plt


# preparation 2: load data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# print(cancer.DESCR)
# print(cancer.feature_names)
# print(cancer.target_name)
# print(cancer.data.shape)

# prepare the train data and test data
from sklearn.model_selection import train_test_split
print(cancer.data)
print(cancer.target)
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state= 42)
# stratify means the same proportions of class labels 

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)



# following is the classification methods like: 
    # logistic regression
    # kNN
    # DT
    # Random forest
    # SVM

# 1 logistic regression
#############################################################################
    
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()
logr.fit(x_train, y_train)
print(logr.score(x_train, y_train))  # score means accuracy
print(logr.score(x_test, y_test))

# improve it ?
logr = LogisticRegression(C=100)  # C is coefficient for cost, smaller means it care more about the majority, default is 1
# the previous model seems to be under fitting
logr.fit(x_train, y_train)
print(logr.score(x_train, y_train))
print(logr.score(x_test, y_test))

print(logr.coef_)
print(logr.intercept_)





# 2 KNN
#############################################################################

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print(knn.score(x_train, y_train))
print(knn.score(x_test, y_test))

# parameter k
acc = []  # Accuracy = True/Total
neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(x_train, y_train)
    acc.append(clf.score(x_train, y_train))





# 3 DT (decision tree)
#############################################################################
    
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))

# feature importance
print(tree.feature_importances_)

# limit the tree depth
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(x_train, y_train)
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))


# (fix it...)
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='cancertree.dot', class_names=['m','b'], feature_names=cancer.feature_names, filled=True)


import pydot
import graphviz
from sklearn.externals.six import StringIO 

dotfile = StringIO()
export_graphviz(tree, out_file=dotfile, class_names=['m','b'], feature_names=cancer.feature_names, filled=True)
pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")

import matplotlib.image as mpimg

img = mpimg.imread('dtree2.png.png')
plt.imshow(img)
plt.show()

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())





# 4 Random Forest
#############################################################################

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(100, n_jobs=-1)
forest.fit(x_train, y_train)

print(forest.score(x_train, y_train))
print(forest.score(x_test, y_test))





# 5 SVM (support vector machine)
#############################################################################

from sklearn.svm import SVC

svm=SVC()
svm.fit(x_train, y_train)
print(svm.score(x_train, y_train))
print(svm.score(x_test, y_test))

# scale the data : see preprocessing
min_train = x_train.min(axis=0)
range_train = (x_train - min_train).max(axis=0)
x_train_scaled = (x_train - min_train)/range_train
x_test_scaled = (x_test - min_train)/range_train

# new train data
svm=SVC()
svm.fit(x_train_scaled, y_train)
print(svm.score(x_train_scaled, y_train))
print(svm.score(x_test_scaled, y_test))


# improve
svm=SVC(C=1400)
svm.fit(x_train_scaled, y_train)
print(svm.score(x_train_scaled, y_train))
print(svm.score(x_test_scaled, y_test))





# 6 others
##############################################################################

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(0.01, probability=True)
gbc.fit(x_train, y_train)
gbc.predict_proba(x_test_scaled[:20])


#  Neuro Network
########################################################################

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
mlp.fit(x_train, y_train)
print(mlp.score(x_train, y_train))
print(mlp.score(x_test, y_test))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

mlp = MLPClassifier(max_iter=2000)
mlp.fit(x_train_scaled, y_train)
print(mlp.score(x_train_scaled, y_train))
print(mlp.score(x_test_scaled, y_test))

# see more about the model
dir(mlp)
mlp

# see the weights
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0], interpolation = 'None', cmap='GnBu')  # GnBu means Green to Blue
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()
plt.show()















# level 2 more about the model
##########################################################################
 
svm.decision_function(x_test_scaled)[:20]  # decision function how certain the classifier is to map the point to possitive class
svm.decision_function(x_test_scaled)[:20] > 0

svm.classes_

svm=SVC(C=1400, probability=True )
svm.fit(x_train_scaled, y_train)

svm.predict(x_test_scaled[:20])
svm.predict_proba(x_test_scaled[:20])


import pickle
pickle.dump(model, 'model.pkl')
model = pickle.load('model.pkl')



# level 3 feature selection
##########################################################################
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
selector = SelectPercentile(0.5)
selector.fit(x_train, y_train)
x_train = selector.transform(x_train)

support = selector.get_support()  # this is a list of boolean showing which is selected

from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(RandomForestClassifier(100), threshold='median')
selector.fit(x_train,y_train)
x_train = selector.transform(x_train)




# level 4 ROC analysis
########################################################################

import numpy as np
from scipy import interp
from itertools import cycle

from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc



# Import some data to play with
iris = load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
from sklearn.multiclass import OneVsRestClassifier

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





'''
model_selection   --    train_test_split
linear_model      --    LogisticRegression
neighbors         --    KNeighborsClassifier
tree              --    DecisionTreeClassifier
ensemble          --    RandomForestClassifier
svm               --    SVC

ensemble          --    GradientBoostingClassifier
multiclass        --    OneVsRestClassifier

neural_network    --    MLPClassifier

'''









''' 

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

data = dfv[attrs].values

data = scale(data)
n_c = 6
model1 = KMeans(n_c)
model1.fit(data)

cluster_dic={0:'good_cast', 1: 'famous_actor_1', 2:'unpopular', 3:'extrodinary_cast', 4:'famous_director', 5: 'normal', np.nan:'extra'}
dfv['fl_cluster'] = [cluster_dic[i] for i in model1.labels_]

'''