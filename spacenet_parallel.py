# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 09:21:51 2015

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

### 2D array to 4D nifti
def array_to_niis(data, mask_img):
    data_ = np.zeros(data.shape[:1] + mask_img.shape)
    data_[:, mask_img.get_data().astype(np.bool)] = data
    data_ = np.transpose(data_, axes=(1,2,3,0))
    return nib.Nifti1Image(data_, mask_img.get_affine())

### main loop
# (train, test, g1_feat, g2_feat, mask_img)
def train_and_test(train, test, g1_feat, g2_feat, mask_img, wprior, lambda_):
    
    x_train_stacked = []
    x_test_stacked = []
    lgr_coeffs = []
    coeffs = []
    for k in range(g1_feat.shape[2]):
        x = np.concatenate((g1_feat[..., k], g2_feat[..., k]), axis=0)
        xtrain = x[train]
        y_train = y[train]
        xtest = x[test]
        y_test = y[test]

        # without prior (spacenet + stacking)
        spnc = SpaceNetClassifier(penalty='smooth-lasso', eps=1e-1,
                                  mask=mask_img, n_jobs=20, memory=CACHE_DIR)
                 
        xtrain_img = array_to_niis(xtrain, mask_img)
        xtest_img = array_to_niis(xtest, mask_img)
        spnc.fit(xtrain_img, y_train)
        x_train_stacked.append(spnc.predict(xtrain_img))
        x_test_stacked.append(spnc.predict(xtest_img))
        coeffs.append(spnc.all_coef_)

    x_train_ = np.asarray(x_train_stacked).T
    x_test_ = np.asarray(x_test_stacked).T
    lgr = LogisticRegression()
    lgr.fit(x_train_, y_train)
    lgr_coeffs = lgr.coef_
    acc = lgr.score(x_test_,  y_test)

    return Bunch(accuracy=acc, coeffs=coeffs, lgr_coeffs=lgr_coeffs)
    
### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_68seeds')

### Load masks
masks = fetch_adni_masks()
mask_img = nib.load(masks['mask_petmr'])

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
X = np.array(X, copy=False)

### load PET prior
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'ad_mci_svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)

### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
y = np.ones(len(idx['AD'][0]) + len(idx_))
y[len(y) - len(g2_feat):] = 0


### prepare shuffle split
n_iter = 1
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))

### spacenet !
for train, test in sss:
    p = train_and_test(train, test, g1_feat, g2_feat, mask_img, w_pet, .7)


"""
from joblib import Parallel, delayed
p = Parallel(n_jobs=40, verbose=5)(delayed(train_and_test)\
(train, test, g1_feat, g2_feat, mask_img, wprior, lambda_) for train, test in sss)
"""

