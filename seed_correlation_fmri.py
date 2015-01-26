# -*- coding: utf-8 -*-
"""
functions for the:
- Extraction of seed of interest
- Computation of the correlations

Created on Wed Jan 21 11:50:49 2015

@author: mehdi.rahim@cea.fr
"""
import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr
from nilearn.input_data import NiftiMasker


def fast_corr(a, b):
    """ fast correlation
        a : time x voxels x [subjects]
    """
    an = a - np.mean(a, axis=0)
    bn = b - np.mean(b, axis=0)
    
    cov = np.sum((an * bn), axis=0)
    
    var = np.sqrt(np.sum(np.square(an), axis=0)) *\
          np.sqrt(np.sum(np.square(bn), axis=0))
    
    return cov / var


### set paths
FIG_PATH = '/disk4t/mehdi/data/tmp/figures'
FMRI_PATH = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features', 'fmri_subjects')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')

### load fmri and seed
dataset = fetch_adni_petmr()
subject_list = dataset['subjects']

cpt = 0
for subj in subject_list:
    fmri = np.load(os.path.join(FMRI_PATH, subj+'.npy'))
    seed = np.load(os.path.join(FMRI_PATH, 'motor_seed_subjects', subj+'.npy'))
    
    corr = []
    for i in np.arange(len(seed)):
        c = fast_corr(fmri, np.tile(seed[i], [fmri.shape[1], 1]).T)
        corr.append(c)
    s_corr = np.mean(corr, axis= 0)
    np.save(os.path.join(FMRI_PATH, 'motor_corr_subjects', subj), s_corr)
    cpt += 1
    print cpt