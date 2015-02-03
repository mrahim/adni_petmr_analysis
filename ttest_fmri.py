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
FIG_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp', 'figures', 'petmr')
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
CORR_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features',
                        'smooth_preproc', 'fmri_subjects')

pet_model_path = os.path.join(FEAT_DIR, 'pet_models', 'svm_coeffs_pet.npz')


### load petmr dataset
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)


### load pet model
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)

### load fMRI features
X = []
for i in np.arange(len(fmri)):
    X.append(np.load(os.path.join(CORR_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)


### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = idx['Normal'][0]
#idx_ = np.hstack((idx['Normal'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
x = x[..., 1]
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

y1 = np.vstack((y, np.abs(y-1))).T

### test
from nilearn.mass_univariate import permuted_ols
pvals, score, fmax = permuted_ols(x, y1, n_jobs=10, verbose=2)

### plot img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
masks = fetch_adni_masks()
masker = NiftiMasker()
masker.mask_img_ = nib.load(masks['mask_petmr'])
img = masker.inverse_transform(pvals[:, 0])
plot_stat_map(img, threshold=2)