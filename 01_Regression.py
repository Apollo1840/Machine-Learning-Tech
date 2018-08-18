# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 23:27:47 2018

@author: zouco

"""

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


# use PolynomialFeatures to impove the performance
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3,include_bias=False)
x_new = poly.fit_transform(x)

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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
model_1 = LinearRegression()
model_1.fit(x_train,y_train)
print(model_1.score(x_train, y_train))  
print(model_1.score(x_test, y_test))

x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.7)
model_2 = LinearRegression()
model_2.fit(x_train,y_train)
print(model_2.score(x_train, y_train))
print(model_2.score(x_test, y_test))