# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:28:45 2015

@author: mr243268
"""

import os
import numpy as np
from sklearn.svm import SVC
from nilearn.input_data import NiftiMasker
from fetch_data import fetch_adni_petmr, fetch_adni_fdg_pet

FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
                         
#dataset = fetch_adni_petmr()
dataset = fetch_adni_fdg_pet()

pet_files = dataset['pet']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)



masker = NiftiMasker(mask_strategy='epi',
                     mask_args=dict(opening=1))
masker.fit(pet_files)
    
if os.path.exists(os.path.join(FEAT_DIR, 'features_voxels_norm_pet.npz')):
    npz = np.load(os.path.join(FEAT_DIR, 'features_voxels_norm_pet.npz'))
    X = npz['X']
    idx = npz['idx'].all()
else:
    pet_masked = masker.transform_imgs(pet_files, n_jobs=4)
    X = np.vstack(pet_masked)
    np.savez(os.path.join(FEAT_DIR, 'features_voxels_norm_pet'), X=X, idx=idx, masker=masker)
    
###

g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

svm = SVC(kernel='linear')
svm.fit(x,y)
coef_map = masker.inverse_transform(svm.coef_)
coef_map.to_filename(os.path.join(FEAT_DIR, 'coef_map_pet.nii.gz'))


