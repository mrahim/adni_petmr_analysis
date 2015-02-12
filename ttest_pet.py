# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 08:46:36 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_petmr, fetch_adni_fdg_pet,\
                       fetch_adni_fdg_pet_diff, fetch_adni_masks
import matplotlib.pyplot as plt

FIG_PATH = '/disk4t/mehdi/data/tmp/figures'

from nilearn.masking import apply_mask
from nilearn.mass_univariate import permuted_ols


###
def ttest_pet(img_files, indices, mask):
    """Returns SVM coeffs learned on PET dataset
    """
    X = apply_mask(img_files, mask)
    
    ### AD / NC classification
    g1_feat = X[idx['AD'][0]]
    idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
    #g2_feat = X[idx['Normal'][0]]
    g2_feat = X[idx_]
        
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0
    
    Y = np.array([y, 1-y]).T

    pvals, tvals , _ = permuted_ols(x, Y, n_perm=10000, two_sided_test=True,
                                    n_jobs=10)

    return pvals, tvals

from fetch_data import set_cache_base_dir, set_features_base_dir

FEAT_DIR = set_features_base_dir() 
PET_DIR = os.path.join(FEAT_DIR, 'pet_models')
CACHE_DIR = set_cache_base_dir()

### Load pet data and mask
mask = fetch_adni_masks()

datasets = {}
#datasets['petmr'] = fetch_adni_petmr()
#datasets['pet_diff']  = fetch_adni_fdg_pet_diff()
datasets['pet'] = fetch_adni_fdg_pet()
all_scores = []

for key in datasets.keys():
    print key
    dataset = datasets[key]
    dx_group = np.array(dataset['dx_group'])
    idx = {}
    for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
        idx[g] = np.where(dx_group == g)
    pvals, tvals = ttest_pet(dataset['pet'], idx, mask['mask_petmr'])
    