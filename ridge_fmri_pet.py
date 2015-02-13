# -*- coding: utf-8 -*-
"""

Ridge regression of fMRI connectivity with nested cross-validation

Created on Wed Feb 11 08:55:37 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from fetch_data import set_features_base_dir, fetch_adni_petmr,\
                        set_cache_base_dir
import matplotlib.pyplot as plt

### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')


### load fMRI features
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(FMRI_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)

### load PET a priori
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'ad_mci_svm_coeffs_petmr.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)

### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

### Stratified Shuffle Split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV, Ridge, RidgeClassifierCV,\
                          LogisticRegression

sss = StratifiedShuffleSplit(y, n_iter=10, test_size=.2)

rdgc_scores = []
rdgp_scores = []
rdgpet_scores = []
yp = []
cpt = 0
for train, test in sss:
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]
    
    xtrain = x_train[..., 0]
    xtest = x_test[..., 0]
    
    # Ridge estimation    
    for lambda_ in [.7]:
        Y = y_train - (lambda_ * np.dot(xtrain, w_pet.T))[:, 0]
        rdg = RidgeCV(alphas=np.logspace(-3, 3, 7))
        rdg.fit(xtrain, Y)
        ### original weights (var substitution)
        wprior = rdg.coef_ - lambda_ * w_pet.T[:, 0]
        
    # Testing
    ## fMRI only
    rdgc = RidgeClassifierCV()
    rdgc.fit(xtrain, y_train)
    rdgc_scores.append(rdgc.score(xtest, y_test))
    
    ## fMRI + PET prior
    rg = Ridge(alpha=rdg.alpha_)
    rg.intercept_ = rdg.intercept_
    rg.coef_ = wprior
    print rg.score(xtest, y_test)
    
    
    y_predict = np.tile(np.dot(xtest, wprior), (1,1)).T
    lgr = LogisticRegression()
    lgr.fit(y_predict, y_test)
    rdgp_scores.append(lgr.score(y_predict, y_test))
    
    ## fMRI on PET model
    y_pet = np.dot(xtest, w_pet.T)
    lgr = LogisticRegression()
    lgr.fit(y_pet, y_test)
    rdgpet_scores.append(lgr.score(y_pet, y_test))
    
    yp.append(y_predict)
    print cpt
    cpt += 1

plt.figure()
plt.boxplot([rdgc_scores, rdgp_scores, rdgpet_scores])