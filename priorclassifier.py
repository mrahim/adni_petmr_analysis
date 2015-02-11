# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:21:34 2015

@author: mehdi.rahim@cea.fr
"""

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

class PriorClassifier:
    """
    Classification with prior integration within regression model
    
    Parameters:
    -----------
    regressor: Estimator
    prior: String or Matrix
    
    
    Attributes:
    coeffs_:  
    scores_: Accuracies
    """
    
    def __init__(self, regressor, prior, lambda_):
        self.regressor = regressor
        self.prior = prior
        self.lambda_ = lambda_
        
    def fit(self, x, y):
        """ Variable transformation """
        Y = y - (self.lambda_ * np.dot(x, self.prior.T))[:, 0]        
        self.regressor.fit(x, Y)
        ### original weights (var substitution)
        self.wprior_ = self.regressor.coef_ - self.lambda_ * self.prior.T[:, 0]        
    
    def score(self, x, y):
        y_predict = np.tile(np.dot(x, self.wprior_), (1,1)).T
        lr = LogisticRegression()
        lr.fit(y_predict, y)
        return lr.score(y_predict, y)