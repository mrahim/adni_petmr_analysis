# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 00:05:40 2015

@author: mr243268
"""

import os
import numpy as np
import fetch_data as data
from sklearn.svm import SVR, SVC
from mord import OrdinalLogistic
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import ShuffleSplit
from nilearn.masking import apply_mask
from sklearn.grid_search import GridSearchCV

FEAT_DIR = data.set_features_base_dir()
CACHE_DIR = data.set_cache_base_dir()

dataset = data.fetch_adni_fdg_pet()
mask = data.fetch_adni_masks()

print "masking"
X = apply_mask(dataset['pet'], mask['mask_petmr'])
Y = np.array(dataset['mmscores'])


idx = data.set_group_indices(dataset['dx_group'])
ind = np.concatenate((idx['AD'][0], idx['Normal'][0]))
dx = np.concatenate((np.ones(len(idx['AD'][0])),
                     np.zeros(len(idx['Normal'][0]))))
"""
x = X[ind,:20:]
x = np.hstack((x,
               np.tile(dx, (1, 1)).T))

y = np.array(Y[ind], dtype=np.int)
#y = y - np.min(y)
"""

x = X[ind, :20:]
y = Y[ind]

ss = ShuffleSplit(len(y), n_iter=1)

score = []
for train, test in ss:
    print 'ss'
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    parameters = {'kernel': ('linear', 'rbf'), 'C': np.logspace(-5,5,51)}
    svr = SVR()
    grd = GridSearchCV(svr, parameters, n_jobs=3)
    grd.fit(x_train, y_train)
    score.append(grd.score(x_test, y_test))
    y_predict = grd.predict(x_test)    
    """
    svr.fit(x_train, y_train)
    score.append(svr.score(x_test, y_test))
    y_predict = svr.predict(x_test)
    """
print score