# -*- coding: utf-8 -*-
"""
Compute PCC correlation features at voxel level on rs-fmri

Created on Tue Jan 20 13:15:00 2015

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


### compute and save correlations on masked fMRI with pet mask
mmasker = NiftiMasker(mask_img=os.path.join(FEAT_DIR, 'pet_mask.nii.gz'))
correlations = []
for i in np.arange(len(func_files)):
    print i
    data = mmasker.fit_transform(func_files[i])
    img = nib.load(func_files[i])
    pcc_values = img.get_data()[26, 22, 28, ...]
    
    corr = []
    for j in np.arange(data.shape[1]):
        corr.append(np.corrcoef(data[:,j], pcc_values)[0,1])
    correlations.append(corr)
    np.savez(os.path.join(FEAT_DIR, 'features_fmri_masked'),
             correlations=correlations, idx=idx)