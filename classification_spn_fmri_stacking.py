# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:56:00 2015

@author: mehdi.rahim@cea.fr
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets.base import Bunch
from nilearn.decoding import SpaceNetRegressor
from fetch_data import fetch_adni_petmr
from fetch_data import set_cache_base_dir, set_features_base_dir,\
                       set_group_indices, array_to_niis, fetch_adni_masks



def train_and_test(X, y, mask, train, test):
    
    x_train_stacked = []
    x_test_stacked = []
    coeffs = []
    y_train, y_test = y[train], y[test]
    for k in range(X.shape[2]):
        x = X[...,k]
        x_train, x_test = x[train], x[test]

        spn = SpaceNetRegressor(penalty='tv-l1', mask=mask,
                                cv=8, max_iter=400)
        
        x_train_img = array_to_niis(x_train, mask)
        spn.fit(x_train_img, y_train)
        x_train_stacked.append(spn.predict(x_train_img))
        x_test_img = array_to_niis(x_test, mask)
        x_test_stacked.append(spn.predict(x_test_img))        
        coeffs.append(spn.coef_)

    x_train_ = np.asarray(x_train_stacked).T[0, ...]
    x_test_ = np.asarray(x_test_stacked).T[0, ...]
    
    lgr = LogisticRegression()
    lgr.fit(x_train_, y_train)
    probas = lgr.decision_function(x_test_)
    scores = lgr.score(x_test_, y_test)
    coeff_lgr = lgr.coef_
    
    B = Bunch(score=scores, proba=probas, coeff=coeffs, coeff_lgr=coeff_lgr)

    ts = str(int(time.time()))
    np.savez_compressed(os.path.join(CACHE_DIR, 'spacenet_stacking_fmri_tv_' + ts),
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
mask = fetch_adni_masks()
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


print 'Classification ...'
n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))
    
from joblib import Parallel, delayed
p = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)(X, y, mask['mask_petmr'], train, test)\
                                    for train, test in sss)

np.savez_compressed(os.path.join(CACHE_DIR, 'spacenet_stacking_fmri_tv_'+str(n_iter)),data=p)
