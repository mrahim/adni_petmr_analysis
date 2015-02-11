# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:54:46 2015

@author: mehdi.rahim@cea.fr
"""

from priorclassifier import PriorClassifier
import os
import numpy as np
from fetch_data import set_features_base_dir, fetch_adni_petmr,\
                        set_cache_base_dir

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV
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
                              'ad_mci_svm_coeffs_pet_diff.npz')
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

sss = StratifiedShuffleSplit(y, n_iter=50, test_size=.2)

scores = []
rdgc_scores = []
rdgp_scores = []
rdgpet_scores = []
yp = []
cpt = 0

for k in range(7):
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
        xtrain = x_train[..., k]
        xtest = x_test[..., k]
    
        rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
        
        pc = PriorClassifier(rdgc, w_pet, .7)
        pc.fit(x[...,k], y)
        rdgc_scores.append(pc.score(xtest, y_test))
    print '--'
    scores.append(rdgc_scores)