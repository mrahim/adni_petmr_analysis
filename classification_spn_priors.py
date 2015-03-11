# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 19:26:42 2015

@author: mr243268
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets.base import Bunch
from nilearn.decoding import SpaceNetRegressor
from fetch_data import fetch_adni_petmr
from fetch_data import set_cache_base_dir, set_features_base_dir,\
                       set_group_indices, array_to_niis, fetch_adni_masks


def train_and_test(X, y, mask, train, test, w_pet, alpha_, lambda_):
    
    x_train_stacked = []
    x_test_stacked = []
    coeffs = []
    y_train, y_test = y, y[test]

    for k in range(X.shape[2]):
        x = X[...,k]
        x_test = x[test]
        xtrain_img = array_to_niis(x, mask)
        xtest_img = array_to_niis(x_test, mask)
        
        spnr = SpaceNetRegressor(penalty='smooth-lasso', w_prior=w_pet,
                                 mask=mask, alpha_=alpha_,
                                 lambda_=lambda_, max_iter=100, cv=4)
        spnr.fit(xtrain_img, y_train)
        x_train_stacked.append(spnr.predict(xtrain_img))
        x_test_stacked.append(spnr.predict(xtest_img))
        coeffs.append(spnr.coef_)

    x_train_ = np.asarray(x_train_stacked).T[0, ...]
    x_test_ = np.asarray(x_test_stacked).T[0, ...]
    
    lgr = LogisticRegression()
    lgr.fit(x_train_, y_train)
    probas = lgr.decision_function(x_test_)
    scores = lgr.score(x_test_, y_test)
    coeff_lgr = lgr.coef_
    
    B = Bunch(score=scores, proba=probas, coeff=coeffs, coeff_lgr=coeff_lgr)

    ts = str(int(time.time()))
    np.savez_compressed(os.path.join(CACHE_DIR, 'spacenet_prior_' + ts),
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
X = np.array(X, copy=False)
y = np.ones(X.shape[0])
y[len(y) - len(idx_):] = 0

### load PET prior
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)[0,:]
alpha_ = .5
lambda_ = 3.7

print 'Classification ...'
n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))

from joblib import Parallel, delayed
p = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)(X, y,
             mask['mask_petmr'], train, test, w_pet, alpha_, lambda_)\
             for train, test in sss)

np.savez_compressed(os.path.join(CACHE_DIR, 'spacenet',
                                 'spacenet_prior_'+str(n_iter)), data=p)