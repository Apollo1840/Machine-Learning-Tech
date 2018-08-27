# -*- coding: utf-8 -*-

# load data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target)


from sklearn import preprocessing
import numpy as np


# for input-------------------------------------------------------
# scale data
x_train_scaled = preprocessing.scale(x_train)

# sometimes we need the scaler for test set and prediction
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # or MinMaxScaler
scaler.fit(x_train)

print(scaler.transform(x_train))

from sklearn.preprocessing import Normalizer
scaler = Normalizer(norm='l1')

# directly transform
print(scaler.transform([[3,2],[6,4]]))


# for output------------------------------------------------------
# 0,1 labeling
# setosa -> 0, versicolor -> 1, virginica -> 2
type_flour = ['setosa', 'versicolor', 'versicolor', 'virginica','virginica', 'setosa', 'versicolor','versicolor']

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(['setosa', 'versicolor', 'virginica'])

print(encoder.transform(type_flour))


from sklearn.preprocessing import LabelBinarizer
lber = LabelBinarizer()
lber.fit(['setosa', 'versicolor', 'virginica'])

print(lber.transform(type_flour))

# another way: 
#   pd.get_dummies(df)
#   pd.get_dummies(df['...'])

# manual classify
data = np.random.rand(3,3)
print(data)

from sklearn.preprocessing import Binarizer
encoder = Binarizer(0.5)
data2 = encoder.transform(data)
print(data2)



##########################################################################
# feature selection

# feature_selection works very like Scaler. First fit it, then transform the data
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest

selector = SelectPercentile(0.5)
selector.fit(x_train, y_train)
x_train = selector.transform(x_train)

support = selector.get_support()  # this is a list of boolean showing which is selected

from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(RandomForestClassifier(100), threshold='1.25*median')
selector.fit(x_train,y_train)
x_train = selector.transform(x_train)



'''
    Excercise:
    1, you need to know following tools, which is in sklearn.preprocessing:
        StandardScaler, 
        MinMaxScaler, 
        Normalizer, 
        LabelEncoder, 
        LabelBinarizer
        Binarizer
        
    2, you need to know how to use those tools:
        .fit, 
        .transform
    
    3, you need to know:
        1) norm in Normalizer
        2) Binarizer(0.5)
    
    4, you need to know following selector, which is in sklearn.feature_selection
        SelectPercentile
        SelectKBest
        SelectFromModel (threshold=)
        
    5, you need to following operations of selector:
        .fit
        .transform
        .fit_transform
        .get_support

'''