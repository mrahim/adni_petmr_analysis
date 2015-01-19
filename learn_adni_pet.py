# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:28:45 2015

@author: mr243268
"""

import os
import numpy as np
from nilearn.input_data import NiftiMasker
from fetch_data import fetch_adni_fdg_pet, fetch_adni_petmr

FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
                         
dataset = fetch_adni_petmr()

pet_files = dataset['pet']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

if os.path.exists(os.path.join(FEAT_DIR, 'features_voxels_norm_petmr.npz')):
    npz = np.load(os.path.join(FEAT_DIR, 'features_voxels_norm_petmr.npz'))
    X = npz['X']
    idx = npz['idx']
else:  
    masker = NiftiMasker(mask_strategy='epi',
                         mask_args=dict(opening=1))
    masker.fit(pet_files)
    pet_masked = masker.transform_imgs(pet_files, n_jobs=4)
    X = np.vstack(pet_masked)
    np.savez(os.path.join(FEAT_DIR, 'features_voxels_norm_petmr'), X=X, idx=idx)
    
###

