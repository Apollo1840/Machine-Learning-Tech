# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# data
x = np.linspace(-10,10,num=30)[:,None]   # [:,None] make the x.shape to be (30,1)
y = -0.1*x + 0.2*x**2 + 0.3*x**3 + 10*np.random.randn(30,1)
plt.plot(x,y,'o')
plt.show()


# linear model
from sklearn.linear_model import LinearRegression
model_1 = LinearRegression()
model_1.fit(x,y)
y_ = model_1.predict(x)
plt.plot(x,y,'o')
plt.plot(x, y_,'r')
plt.show()

print(model_1.coef_)
print(model_1.intercept_)

# l2 loss
from sklearn.metrics import mean_square_error
y_pred = model_1.predict(x)
print(mean_square_error(y_pred, y))

# use PolynomialFeatures to impove the performance
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3,include_bias=False)  # bias means columns with all 1
x_new = poly.fit_transform(x)


# PolynomialFeatures for multiple column:
poly = PolynomialFeatures(2,include_bias=False) # bias means columns with all 1
print(poly.fit_transform([[1,5,2],[3,10,4]])) # a, b, c, a**2, ab, ac, b**2, bc, c**2

poly = PolynomialFeatures(3,include_bias=False) 
print(poly.fit_transform([[1,5,2],[3,10,4]])) # a, b, c, a**2, ab, ac, b**2, bc, c**2, a**3, a**2b, a**2c, ab**2c, abc, ac**2, ... 

# how to use interaction_only
poly = PolynomialFeatures(2,include_bias=False, interaction_only=True)  
print(poly.fit_transform([[1,5,2],[3,10,4]])) # a, b, c, ab, ac, bc

poly = PolynomialFeatures(3,include_bias=False, interaction_only=True)  
print(poly.fit_transform([[1,5,2,1],[3,10,4,1]])) # a, b, c, d, ab, ac, ad, bc, bd, cd, abc, abd, acd, bcd


# building polynomial regression by this PolynomialFeatures
model_2 = LinearRegression(fit_intercept=False)  # assume we know y=0 when x=0
model_2.fit(x_new,y)
y_ = model_2.predict(x_new)
plt.plot(x,y,'o')
plt.plot(x, y_,'r')
plt.show()

print(model_2.coef_)
print(model_2.intercept_)


# why we say model_2 is better than model_1:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7) # be careful of the order
model_1 = LinearRegression()
model_1.fit(x_train,y_train)
print(model_1.score(x_train, y_train))  
print(model_1.score(x_test, y_test))

model_2 = LinearRegression()
model_2.fit(x_train,y_train)
print(model_2.score(x_train, y_train))
print(model_2.score(x_test, y_test))


# more about the score:
# the score is r2_score, it can also calculate in this way:
from sklearn.metrics import r2_score
print(r2_score(model_1.predict(x_train), y_train))


# for polynomial_regression(pr) use pipeline to create some kind of network:
from sklearn.pipeline import Pipeline
pr3=Pipeline([('Poly', PolynomialFeatures(3,include_bias=False)), ('lm', LinearRegression())])

pr3.fit(x, y)
y_ = pr3.predict(x)
plt.plot(x,y,'o')
plt.plot(x, y_,'r')
plt.show()

print(pr3.named_steps['lm'].coef_)
print(pr3.named_steps['lm'].intercept_)








'''
experience: 
    
    1. regularization is not the gold finger for overfitting problem. 
        sometimes the model is too bad, so that it is really hard to find the suitable alpha value. 
        it is either too small or too big.
        
    2. overfitting often occurs when data amount is less. and the variance is relatively big.
    
'''

# linear model with penalty on weights (regularization)
x = np.linspace(-10,10,num=20)[:,None]   # [:,None] make the x.shape to be (30,1)
y = -0.1*x + 0.2*x**2 + 0.3*x**3 + 20*np.random.randn(20,1)

ploy = PolynomialFeatures(10, include_bias=False)  
x_new = ploy.fit_transform(x)

x_new = x_new*(10**(-3)) # this step is important

x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.7)

model_overfit = LinearRegression(fit_intercept=False)  # assume we know y=0 when x=0
model_overfit.fit(x_train,y_train)

print(model_overfit.score(x_train, y_train))
print(model_overfit.score(x_test, y_test))

y_ = model_overfit.predict(x_new)
plt.plot(x,y,'o')
plt.plot(x, y_,'r')
plt.show()


from sklearn.linear_model import Ridge
model = Ridge(alpha=20, fit_intercept=False)  # assume we know y=0 when x=0
model.fit(x_train,y_train)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

y_ = model.predict(x_new)
plt.plot(x,y,'o')
plt.plot(x, y_,'r')
plt.show()






'''
    Excercise:
        1, you need to know following model in sklearn.linear_model:
            LinearRegression
            Ridge
        
        2, you need to following functions and attributes of a model:
            .fit
            .predict
            .score
            .coef_
            .intercept_
        
        3, you need to know following param in the model:
            fit_intercept in LinearRegression
            alpha in Ridge
            
        4, you need to know how to use PolynomialFeatures in sklearn.preprocessing
            include_bias =
            interaction_only = 
        
            .fit_transform
            
        5, you need to know how to use polynomial regression and how to use Pipeline.
        
        6, you need to know how to use train_test_split
            train_size = 
            
        7, how to score the model (sklearn.metrics)
            mean_square_error()
            r2_error()
        

'''


# robust regression
from sklearn.datasets import make_regression

# data
n_samples = 1000
X, y, coef = make_regression(n_samples=n_samples,
                                      n_features=1,
                                      n_informative=1,
                                      noise=10,
                                      coef=True,
                                      random_state=0)

n_outliers = 50
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 0.5 * np.random.normal(size=n_outliers)


model = LinearRegression()
model.fit(X, y)

from sklearn.linear_model import RANSACRegressor
model_ransac = RANSACRegressor(LinearRegression())

model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


line_X = np.arange(-5, 5)
line_y = model.predict(line_X[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])


plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(line_X, line_y, '-k', label='Linear Regression')
plt.plot(line_X, line_y_ransac, '-b', label="RANSAC Regression")
plt.legend(loc='upper left')

plt.show()

