# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:55:02 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from fetch_data import fetch_adni_petmr
from sklearn.cross_validation import StratifiedShuffleSplit
from nilearn.plotting import plot_stat_map
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC

FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
CORR_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features',
                        'fmri_subjects', 'corr_subjects')

### load mask, features, 
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])

idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

### load features
X = []
for i in np.arange(len(fmri)):
    X.append(np.load(os.path.join(CORR_DIR, subj_list[i]+'.npy')))
X = np.array(X)

groups = [['AD', 'Normal'], ['AD', 'EMCI']]
coeffs = []
for gr in groups:
    g1_feat = X[idx[gr[0]][0]]
    idx_ = idx[gr[1]][0]
    g2_feat = X[idx_]
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0

    sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)
    
    cpt = 0
    coeff = []
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        coeff.append(svm.coef_)
        cpt += 1
        print cpt
    coeffs.append(np.mean(coeff, axis=0))
    masker = NiftiMasker()
    masker.mask_img_ = nib.load(os.path.join(FEAT_DIR, 'pet_mask.nii.gz'))
    img = masker.inverse_transform(np.mean(coeff, axis=0))
    img.to_filename(os.path.join(FEAT_DIR, 'niis',
                                 'map_fmri_'+'_'.join(gr)+'.nii.gz'))
    plot_stat_map(img, threshold=0.0004, title='+'.join(gr)+' weights',
                  output_file='map_fmri_'+'_'.join(gr))
