# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import set_cache_base_dir, fetch_adni_petmr
from fetch_data import set_features_base_dir
import matplotlib.pyplot as plt

FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')
CACHE_DIR = set_cache_base_dir()
FIG_PATH = os.path.join(CACHE_DIR, 'figures')



dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])

idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)


### load fMRI features
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(FMRI_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)


groups = [['AD' , 'Normal']]
scores = []
coeffs = []
for k in range(X.shape[2]):
    score = []
    for gr in groups:
        g1_feat = X[idx[gr[0]][0]]
        idx_ = idx[gr[1]][0]
        idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
        g2_feat = X[idx_]
        x = np.concatenate((g1_feat[...,k], g2_feat[...,k]), axis=0)
        y = np.ones(len(x))
        y[len(x) - len(g2_feat):] = 0
        
        sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)
        
        cpt = 0
        score = []
        coeff = []
        for train, test in sss:
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            
            svm = SVC(kernel='linear')
            svm.fit(x_train, y_train)
            score.append(svm.score(x_test, y_test))
            coeff.append(svm.coef_)
            cpt += 1
            print k, cpt
    scores.append(score)
    coeffs.append(np.mean(coeff, axis=0))

"""
### SVM coeffs
for k in range(X.shape[2]):
    np.savez_compressed(os.path.join(FEAT_DIR, 'fmri_models',
                                     'svm_coeffs_fmri_seed_'+str(k)),
                        svm_coeffs=coeffs[k],
                        idx=idx,
                        subjects=dataset['subjects'])
"""