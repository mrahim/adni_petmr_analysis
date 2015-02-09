# -*- coding: utf-8 -*-
"""

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_petmr, fetch_adni_masks

### set paths
BASE_DIR = os.path.join('/', 'disk4t', 'mehdi')
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.join('/', 'home', 'mr243268')

FIG_DIR = os.path.join(BASE_DIR, 'data', 'tmp', 'figures', 'petmr')
FEAT_DIR = os.path.join(BASE_DIR, 'data', 'features')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'tmp')
CORR_DIR = os.path.join(BASE_DIR, 'data', 'features',
                        'smooth_preproc', 'fmri_subjects')

pet_model_path = os.path.join(FEAT_DIR, 'pet_models', 'svm_coeffs_pet.npz')


### load petmr dataset

dataset = np.load('/home/mr243268/data/features/fmri_models/svm_coeffs_fmri_seed_0.npz')
subj_list = dataset['subjects']
idx =  np.any(dataset['idx'])
"""

dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)
"""

### load pet model
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)

### load fMRI features
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(CORR_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)


### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['Normal'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
x = x[..., 0]
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

y1 = np.vstack((y, np.abs(y-1))).T

### test
"""
from nilearn.mass_univariate import permuted_ols
pvals, score, fmax = permuted_ols(x, y1, n_jobs=1, verbose=2)
"""


seeds = ['PCC', 'Visual', 'Motor', 'ACC', 'L-Parietal', 'R-Parietal', 'Prefrontal']

from scipy import stats
for k in range(7):

    tvals, pvals = stats.ttest_ind(g1_feat[...,k], g2_feat[...,k])
    pvals = - np.log10(pvals)
    
    ### plot img
    from nilearn.input_data import NiftiMasker
    from nilearn.plotting import plot_stat_map
    masks = fetch_adni_masks()
    masker = NiftiMasker()
    masker.mask_img_ = nib.load(masks['mask_petmr'])
    img = masker.inverse_transform(tvals)
    plot_stat_map(img, threshold=2, title=seeds[k])