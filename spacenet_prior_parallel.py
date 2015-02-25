# -*- coding: utf-8 -*-
"""
Spacenet using the prior

Created on Wed Feb 11 15:54:46 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import set_features_base_dir, fetch_adni_petmr,\
                        set_cache_base_dir, fetch_adni_masks, set_group_indices
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from nilearn.decoding import SpaceNetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.base import Bunch


### 2D array to 4D nifti
def array_to_niis(data, mask_img):
    data_ = np.zeros(data.shape[:1] + mask_img.shape)
    data_[:, mask_img.get_data().astype(np.bool)] = data
    data_ = np.transpose(data_, axes=(1,2,3,0))
    return nib.Nifti1Image(data_, mask_img.get_affine())

def train_and_test(train, test, g1_feat, g2_feat, mask_img, w_pet,
                   alpha_, lambda_):
    
    x_train_stacked = []
    x_test_stacked = []
    lgr_coeffs = []
    coeffs = []
    for k in range(g1_feat.shape[2]):

        x = np.concatenate((g1_feat[..., k], g2_feat[..., k]), axis=0)
        xtrain, y_train = x[train], y[train]
        xtrain_img = array_to_niis(xtrain, mask_img)
        xtest, y_test = x[test], y[test]
        xtest_img = array_to_niis(xtest, mask_img)
        
        spnr = SpaceNetRegressor(penalty='smooth-lasso', w_prior=w_pet,
                                 mask=mask_img, alpha_=alpha_,
                                 lambda_=lambda_, max_iter=100, cv=4)
        spnr.fit(xtrain_img, y_train)
        x_train_stacked.append(spnr.predict(xtrain_img))
        x_test_stacked.append(spnr.predict(xtest_img))
        coeffs.append(spnr.coef_)

    x_train_ = np.asarray(x_train_stacked).T
    x_test_ = np.asarray(x_test_stacked).T
    lgr = LogisticRegression()
    lgr.fit(x_train_, y_train)
    lgr_coeffs = lgr.coef_
    acc = lgr.score(x_test_, y_test)

    return Bunch(accuracy=acc, coeffs=coeffs, lgr_coeffs=lgr_coeffs)


### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_68seeds')


### load fMRI features
masks = fetch_adni_masks()
mask_img = nib.load(masks['mask_petmr'])
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = set_group_indices(dx_group)    
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(FMRI_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X, copy=False)

### load PET a priori
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)
w_pet = w_pet[0,:]
#w_pet = w_pet/np.max(w_pet)

### prepare data
g1_feat = X[idx['AD'][0]]
#idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
y = np.ones(len(idx['AD'][0]) + len(idx_))
y[len(y) - len(g2_feat):] = 0


n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))

lambda_ = 3.7
alpha_ = .5
"""
for train, test in sss:
    p = train_and_test(train, test, g1_feat, g2_feat,
                       mask_img, w_pet, alpha_, lambda_)
"""
from joblib import Parallel, delayed
p = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)\
(train, test, g1_feat, g2_feat, mask_img) for train, test in sss)

np.savez_compressed('spacenet_prior_sl_'+str(n_iter),data=p)
