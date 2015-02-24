# -*- coding: utf-8 -*-
"""
    Prior integration with the spatial regularization
    
    1. Load X, y, w_prior, alpha, lambda
    2. Compute ~X, ~y
    3. Solve spacenet

Created on Mon Feb 23 14:40:32 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr, fetch_adni_masks,\
                       set_features_base_dir, set_cache_base_dir
import matplotlib.pyplot as plt

from nilearn.decoding import SpaceNetClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.base import Bunch


### 0. Set paths, load masks
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_68seeds')

masks = fetch_adni_masks()
mask_img = nib.load(masks['mask_petmr'])

### 1. Load X, y, w_prior, alpha, lambda
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
X = np.array(X, copy=False)
g1_feat = X[idx['AD'][0]]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
y = np.ones(len(idx['AD'][0]) + len(idx_))
y[len(y) - len(g2_feat):] = 0

pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'ad_mci_svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_prior = np.array(model)

alpha_ = 1
lambda_ = .7

### 2. Compute ~X and ~y
from scipy import linalg
x = X[...,0]
print 'X_tilde'
X_tilde = linalg.cholesky(np.dot(x, x.T)+np.eye(x.shape[1]))
print 'y_tilde'
X_tilde_inv = linalg.inv(X_tilde)
y_tilde = np.dot(X_tilde_inv, (np.dot(X,y) + lambda_*w_prior))


