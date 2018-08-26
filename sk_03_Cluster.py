# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.

from sklearn.cluster import KMeans

n_c = 3
model = KMeans(n_c)
model.fit(X)

print(model.labels_)

print(model.cluster_centers_)

print(model.score(labels_true, model.labels_))  # this is AMI adjusted mutual information

# Plot the result
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, edgecolors='k', cmap=plt.cm.Paired)

# multiple assignment
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()


from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(3)
model.fit(X)


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(3) 
model.fit(X)

model = AgglomerativeClustering(3, linkage='ward')  # ward is default
# ward : minimizes the variance of the clusters being merged.
# average : uses the average of the distances of each observation of the two sets.
# complete : the maximum distances between all observations of the two sets.
# single : in Scipy

model.fit(X)



from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.3, min_samples=5,metric='euclidean', n_jobs=-1)
model.fit(X)

from sklearn.mixture import GMM
model = GMM(3)
model.fit(X)


'''
    1, you should know the clusters:
        sklearn.cluster      --     KMeans
        sklearn.cluster      --     MiniBatchKMeans
        sklearn.cluster      --     AgglomerativeClustering
        sklearn.cluster      --     DBSCAN
        sklearn.mixture      --     GMM
    
    2, you should know properties of the model
        .fit
        .labels_
        .cluster_centers_
        .score()


'''