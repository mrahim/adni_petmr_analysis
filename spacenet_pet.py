# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:17:30 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_fdg_pet_diff, set_cache_base_dir,\
                        set_features_base_dir, fetch_adni_masks,\
                        set_group_indices
from nilearn.decoding import SpaceNetClassifier

CACHE_DIR = set_cache_base_dir()
FEAT_DIR = set_features_base_dir()

dataset = fetch_adni_fdg_pet_diff()
idx = set_group_indices(dataset['dx_group'])
pet_files = dataset['pet']

mask = fetch_adni_masks()
mask_img = nib.load(mask['mask_petmr'])


"""
spnc = SpaceNetClassifier(penalty='smooth-lasso', eps=1e-1,
                          mask=mask_img, n_jobs=20, memory=CACHE_DIR)
                          
spnc.fit(X, y)
"""