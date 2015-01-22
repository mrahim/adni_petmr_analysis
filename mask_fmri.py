# -*- coding: utf-8 -*-
"""

Created on Wed Jan 21 15:00:44 2015

@author: mehdi.rahim@cea.fr
"""


import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr
from nilearn.input_data import NiftiMasker

### set paths
FIG_PATH = '/disk4t/mehdi/data/tmp/figures'
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')

### fetch fmri
dataset = fetch_adni_petmr()
func_files = dataset['func']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)


mask_img = nib.load(os.path.join(FEAT_DIR, 'pet_mask.nii.gz'))

from nilearn.masking import apply_mask
from nilearn.image import resample_img
data = []
cpt = 0
for f in func_files:
    img = resample_img(f, target_affine=mask_img.get_affine(),
                       target_shape=mask_img.get_shape())
    data.append(apply_mask(img, mask_img))
    cpt += 1
    print cpt