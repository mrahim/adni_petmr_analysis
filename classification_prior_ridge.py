# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:05:20 2015

@author: mehdi.rahim@cea.fr
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from priorclassifier import PriorClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets.base import Bunch
from fetch_data import fetch_adni_petmr
from fetch_data import set_cache_base_dir, set_features_base_dir,\
                       set_group_indices


def train_and_test(X, y, train, test, w_pet, lambda_):
    
    x_train_stacked = []
    x_test_stacked = []
    coeffs = []
    y_train, y_test = y[train], y[test]
    for k in range(X.shape[2]):
        x = X[...,k]
        x_train, x_test = x[train], x[test]

        rdg = RidgeCV(alphas=np.logspace(-3, 3, 7))
        rdg.fit(x_train, y_train)
        pc = PriorClassifier(rdg, w_pet, lambda_)
        pc.fit(x[:,...], y[:])

        x_train_stacked.append(pc.predict(x_train))
        x_test_stacked.append(pc.predict(x_test))
        coeffs.append(rdg.coef_)

    x_train_ = np.asarray(x_train_stacked).T[0, ...]
    x_test_ = np.asarray(x_test_stacked).T[0, ...]
    
    lgr = LogisticRegression()
    lgr.fit(x_train_, y_train)
    probas = lgr.decision_function(x_test_)
    scores = lgr.score(x_test_, y_test)
    coeff_lgr = lgr.coef_
    
    B = Bunch(score=scores, proba=probas, coeff=coeffs, coeff_lgr=coeff_lgr)

    ts = str(int(time.time()))
    np.savez_compressed(os.path.join(CACHE_DIR, 'ridge_prior_' + ts),
                        data=B)
    return B


###########################################################################
###########################################################################

### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_68seeds')


### load dataset
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
idx = set_group_indices(dataset['dx_group'])
idx_ = np.hstack((idx['EMCI'][0], idx['LMCI'][0]))
img_idx = np.hstack((idx['AD'][0], idx_))
X = []
print 'Loading data ...'
for i in img_idx:
    X.append(np.load(os.path.join(FMRI_DIR, subj_list[i]+'.npz'))['corr'])
# X.shape = (n_samples, n_features, n_rois)
X = np.array(X)
y = np.ones(X.shape[0])
y[len(y) - len(idx_):] = 0


### load PET a priori
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)

print 'Classification ...'
n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))

from joblib import Parallel, delayed
p = Parallel(n_jobs=10, verbose=5)(delayed(train_and_test)(X, y, train, test,
             w_pet, .7) for train, test in sss)

np.savez_compressed(os.path.join(CACHE_DIR, 'ridge_prior_'+str(n_iter)),data=p)
