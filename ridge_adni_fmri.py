# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:15:00 2015

@author: mr243268
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr

FIG_PATH = '/disk4t/mehdi/data/tmp/figures'
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')

dataset = fetch_adni_petmr()


# PET 
npz = np.load(os.path.join(FEAT_DIR, 'features_voxels_norm_pet.npz'))
X = npz['X']
idx = npz['idx'].all()
masker = npz['masker']


# FMRI
func_files = dataset['func']
from nilearn.input_data import NiftiMasker

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

    np.savez(os.path.join(FEAT_DIR, 'features_fmri_masked'), correlations=correlations)

"""

###

g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

###

from sklearn.linear_model import RidgeCV

rdg = RidgeCV(alphas=np.logspace(-3,3,7), cv=None, fit_intercept=True,
              scoring=None, normalize=False)
rdg.fit(x,y)

"""