# -*- coding: utf-8 -*-
"""
Script that computes ridge coeffs on PET voxels
Created on Mon Jan 26 11:16:46 2015

@author: mehdi.rahim@cea.fr
"""

import numpy as np
import nibabel as nib
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_fdg_pet, fetch_adni_masks, \
                       set_cache_base_dir, set_group_indices
from nilearn.masking import apply_mask
from sklearn.datasets.base import Bunch
from sklearn.linear_model import RidgeClassifierCV

def train_and_test(X, y, train, test):
    """Returns a bunch containing the score,
    the proba and the coeff of the train and test iteration.
    """
    x_train, y_train = X[train], y[train]
    x_test, y_test = X[test], y[test]

    rdgc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    rdgc.fit(x_train, y_train)
    proba = rdgc.decision_function(x_test)
    score = rdgc.score(x_test, y_test)
    coeff = rdgc.coef_
    return Bunch(score=score, proba=proba, coeff=coeff)

#################################################
CACHE_DIR = set_cache_base_dir()

### Load pet data and mask
mask = fetch_adni_masks()
dataset = fetch_adni_fdg_pet()
pet_files = np.array(dataset['pet'])
idx = set_group_indices(dataset['dx_group'])
idx_ = np.hstack((idx['Normal']))
img_idx = np.hstack((idx['AD'], idx_))

### Mask data
mask_img = nib.load(mask['mask_petmr'])
X = apply_mask(pet_files[img_idx], mask_img)
y = np.ones(X.shape[0])
y[len(y) - len(idx_):] = 0

n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))

from joblib import Parallel, delayed
p = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)\
(X, y, train, test) for train, test in sss)

np.savez_compressed('ridge_classif_pet_'+str(n_iter), data=p)
