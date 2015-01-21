# -*- coding: utf-8 -*-
"""
Compute a voxel-based model of AD/NC classification on PET images

Created on Mon Jan 19 14:28:45 2015
@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from sklearn.svm import SVC
from nilearn.input_data import NiftiMasker
from fetch_data import fetch_adni_fdg_pet


### set paths
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
feat_path = os.path.join(FEAT_DIR, 'features_voxels_norm_pet.npz')

# fetch and mask pet images
dataset = fetch_adni_fdg_pet()
pet_files = dataset['pet']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)
masker = NiftiMasker(mask_strategy='epi',
                     mask_args=dict(opening=1))
masker.fit(pet_files)


### load or compute features
if os.path.exists(feat_path):
    npz = np.load(feat_path)
    X = npz['X']
    idx = npz['idx'].all()
else:
    pet_masked = masker.transform_imgs(pet_files, n_jobs=4)
    X = np.vstack(pet_masked)
    np.savez(feat_path, X=X, idx=idx, masker=masker)
    
### prepare data for classification
g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0


### compute and save a linear SVC model
coeffs = []
scores = []
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)

cpt = 0
for train, test in sss:
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]
    
    svm = SVC(kernel='linear')
    svm.fit(x_train, y_train)
    scores.append(svm.score(x_test, y_test))
    coeffs.append(svm.coef_)
    cpt += 1
    print cpt

c = np.mean(np.vstack(np.array(coeffs)), axis=0)
np.savez(os.path.join(FEAT_DIR, 'coef_mean_map_pet'),
         coef_map=c, idx=idx, masker=masker)
"""
coef_map = masker.inverse_transform(svm.coef_)
coef_map.to_filename(os.path.join(FEAT_DIR, 'coef_map_pet.nii.gz'))
np.savez(os.path.join(FEAT_DIR, 'coef_map_pet'),
         coef_map=svm.coef_, idx=idx, masker=masker)
"""
