# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 08:44:38 2018

@author: zouco
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    train_upper_level = train_scores_mean + train_scores_std
    train_lower_level = train_scores_mean - train_scores_std
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    test_upper_level = test_scores_mean + test_scores_std
    test_lower_level = test_scores_mean - test_scores_std

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("number_samples")
        plt.ylabel("score")
        plt.gca().invert_yaxis()
        plt.grid()
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="train score")
        plt.fill_between(train_sizes, train_lower_level, train_upper_level, alpha=0.1, color="b")
        
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="validation score")
        plt.fill_between(train_sizes, test_lower_level, test_upper_level, alpha=0.1, color="r")
        
        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

# plot_learning_curve(lr, "learning curve", x_train, y_train)

# help(learning_curve)