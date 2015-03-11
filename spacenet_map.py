# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:53:17 2015

@author: mehdi.rahim@cea.fr
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets.base import Bunch
from nilearn.decoding import SpaceNetClassifier
from fetch_data import fetch_adni_petmr
from fetch_data import set_cache_base_dir, set_features_base_dir,\
                       set_group_indices, array_to_niis, fetch_adni_masks



def train(X, y, mask):
    
    for k in [55]:
        x = X[...,k]

        spn = SpaceNetClassifier(penalty='smooth-lasso', mask=mask,
                                cv=8, max_iter=1000, n_jobs=20)
        x_img = array_to_niis(x, mask)
        spn.fit(x_img, y)
        coeffs = spn.coef_

    np.savez_compressed(os.path.join(CACHE_DIR, 'spacenet_stacking_fmri_sl_55'),
                        coeffs=coeffs)
    return spn

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

p = train(X, y, mask['mask_petmr'])

np.savez_compressed(os.path.join(CACHE_DIR, 'spacenet_stacking_fmri_sl_total'),
                    data=p)
