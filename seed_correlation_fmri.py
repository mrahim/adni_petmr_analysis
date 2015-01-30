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
from joblib import Parallel, delayed


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

###
def seed_correlation_subjects(subject_list, subject_folder):
    output_folder = 'corr_' + subject_folder
    if not os.path.isdir(os.path.join(FMRI_PATH, output_folder)):
        os.mkdir(os.path.join(FMRI_PATH, output_folder))
    cpt = 0
    for subj in subject_list:
        fmri = np.load(os.path.join(FMRI_PATH, subj+'.npy')).T
        seed = np.load(os.path.join(FMRI_PATH, subject_folder, subj+'.npy'))
        a = np.tile(np.tile(fmri, (1,1,1)), (seed.shape[0],1,1))
        b = np.tile(np.tile(seed, (1,1,1)), (fmri.shape[0],1,1))
        a = np.transpose(a, (2,0,1))
        b = np.transpose(b, (2,1,0))
              
        p = Parallel(n_jobs=10, verbose=2)(delayed(fast_corr)(a[:,:,i], b[:,:,i]) for i in range(a.shape[2]))
        s_corr = np.mean(p, axis=1)
    
        np.save(os.path.join(FMRI_PATH, output_folder, subj), s_corr)
        cpt += 1
        print cpt,'/',len(subject_list)


### set paths
FIG_PATH = '/disk4t/mehdi/data/tmp/figures'
FMRI_PATH = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features', 'fmri_subjects')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')

### load fmri and seed
dataset = fetch_adni_petmr()
subject_list = dataset['subjects']

seed_correlation_subjects(subject_list, 'visual_seed_subjects')

"""
cpt = 0
for subj in subject_list:
    fmri = np.load(os.path.join(FMRI_PATH, subj+'.npy')).T
    seed = np.load(os.path.join(FMRI_PATH, 'motor_seed_subjects', subj+'.npy'))
    a = np.tile(np.tile(fmri, (1,1,1)), (len(seed),1,1))
    b = np.tile(np.tile(seed, (1,1,1)), (232073,1,1))
    a = np.transpose(a, (2,0,1))
    b = np.transpose(b, (2,1,0))
    s_corr = np.mean(fast_corr(a,b), axis=0)

    np.save(os.path.join(FMRI_PATH, 'motor_corr_subjects', subj), s_corr)
    cpt += 1
    print cpt
"""