# -*- coding: utf-8 -*-
"""
Compute a ridge that combines PET model and fMRI correlations.
The general formula is :
|Xw - y|^2 + alpha |w - lambda w_tep|^2

By making :
beta = w - lambda w_tep

We have :
|X beta - (y - lambda X w_tep)|^2 + alpha |beta|^2


Created on Wed Jan 21 09:05:28 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_petmr
from fetch_data import set_cache_base_dir, set_features_base_dir




def ridge_apriori(a, y, w_pet, lambda_=.7, n_iter=100):
    
    y_predict = []
    for k in range(a.shape[2]):
        x = a[...,k]
        Y = y - lambda_ * np.dot(x, w_pet.T)
        rdg = RidgeCV(alphas=np.logspace(-3, 3, 70))
        X = x
        rdg.fit(X,Y)    
        ### original weights (var substitution)
        w = rdg.coef_[0, :].T - lambda_ * w_pet
        # y_predict is the regression result of the ridge+PETapriori
        y_predict.append(np.dot(X, w.T))
        
        
    ### ShuffleSplit comparison
    ss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2)
    fmri_predict = []
    rdg_predict = []
    cpt = 0
    for train, test in ss:
        fmri_pr = []
        rdg_pr = []
        for k in range(a.shape[2]):
            x = a[...,k]
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            lgr = LogisticRegression()
            lgr.fit(x_train, y_train)
            fmri_pr.append(lgr.predict_proba(x_test)[0, :])
        
            y_p = y_predict[k][test]
            lgr = LogisticRegression()
            lgr.fit(y_p, y_test)
            rdg_pr.append(lgr.predict_proba(y_p)[0, :])

        rdg_pr = np.array(rdg_pr)
        lgr = LogisticRegression()
        lgr.fit(rdg_pr, y_test)
        print lgr.score(rdg_pr, y_test)
        
        
        fmri_predict.append(fmri_pr)
        rdg_predict.append(rdg_pr)
        cpt += 1
        print cpt
    return fmri_predict, rdg_predict

### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')


### load fMRI features
#corr_path = os.path.join(FEAT_DIR, 'features_fmri_masked.npz')
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


### load pet model
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'ad_mci_svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']

### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

a = np.copy(x)
w_pet = np.array(model)

### Ridge with variable substitution
fmri_proba, rdg_proba = ridge_apriori(a, y, w_pet, lambda_=.7, n_iter=5)