# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:17:30 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_fdg_pet_diff, set_cache_base_dir,\
                        set_features_base_dir, fetch_adni_masks,\
                        set_group_indices
from nilearn.decoding import SpaceNetClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.datasets.base import Bunch

####

def train_and_test(X, y, spnc, train, test):

    xtrain = X[train]
    ytrain = y[train]
    xtest = X[test]
    ytest = y[test]    
    spnc.fit(xtrain, ytrain)
    ypredict = spnc.predict(xtest)
    score = accuracy_score(ytest, ypredict)
    coeff = spnc.coef_

    return Bunch(score=score, coeff=coeff)

####

CACHE_DIR = set_cache_base_dir()
FEAT_DIR = set_features_base_dir()

dataset = fetch_adni_fdg_pet_diff()
idx = set_group_indices(dataset['dx_group'])
pet_files = dataset['pet']

X = pet_files[idx['AD'][0]]
X = np.concatenate((X, pet_files[idx['Normal'][0]]))


y = np.ones(len(X))
y[len(X) - len(pet_files[idx['Normal'][0]]):] = 0


mask = fetch_adni_masks()
mask_img = nib.load(mask['mask_petmr'])

spnc = SpaceNetClassifier(penalty='smooth-lasso', eps=1e-1,
                          mask=mask_img)

n_iter = 50
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2)
score = []
coeff = []


from joblib import Parallel, delayed
p = Parallel(n_jobs=10, verbose=5)(delayed(train_and_test)\
(X, y, spnc, train, test) for train, test in sss)

np.savez_compressed('spacenet_pet_sl_50', data=p)

"""
for train, test in sss:
    xtrain = X[train]
    ytrain = y[train]
    xtest = X[test]
    ytest = y[test]    
    spnc.fit(xtrain, ytrain)
    ypredict = spnc.predict(xtest)
    score.append(accuracy_score(ytest, ypredict))
    coeff.append(spnc.coef_)
"""