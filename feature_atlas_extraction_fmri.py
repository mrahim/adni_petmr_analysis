# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 15:24:31 2015

@author: mehdi.rahim@cea.fr
"""


import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr, fetch_adni_masks
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.datasets import fetch_msdl_atlas

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


def fmri_connectivity(func_file, img_masker, map_masker, subject_id):
    ### 1-Extract seeds
    ### 2-Extract fMRI voxels
    ### 3-Compute correlations for each voxel

    if not os.path.isfile(os.path.join(FMRI_DIR, subject_id + '.npz')):
        
        map_values = map_masker.fit_transform(func_file)
        fmri_values = img_masker.transform(func_file)
        
        n_map = map_values.shape[1]
        n_voxels = fmri_values.shape[1]
        
        c = []
        for i in range(n_voxels):
            fmri_vox = np.tile(fmri_values[:, i], (39, 1)).T
            c.append(fast_corr(map_values, fmri_vox))
        np.savez_compressed(os.path.join(FMRI_DIR, subject_id),
                            corr=np.array(c))

#######################################################
#######################################################

### set paths
from fetch_data import set_cache_base_dir, set_features_base_dir
CACHE_DIR = set_cache_base_dir()
FIG_PATH = os.path.join(CACHE_DIR, 'figures')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_msdl_atlas')

### fetch fmri, load masks and seeds
dataset = fetch_adni_petmr()
func_files = dataset['func']
subject_list = dataset['subjects']
mask = fetch_adni_masks()

### Labels

atlas = fetch_msdl_atlas()
mmasker = NiftiMapsMasker(maps_img=atlas['maps'], mask_img=mask['mask_petmr'],
                            resampling_target='data', detrend=True,
                            standardize=False, t_r=3.)
mmasker.maps_img_ = nib.load(atlas['maps'])
mmasker.mask_img_ = nib.load(mask['mask_petmr'])

### fMRI
fmasker = NiftiMasker(mask_img=mask['mask_petmr'], detrend=True,
                      standardize=False, t_r=3.)
fmasker.mask_img_ = nib.load(mask['mask_petmr'])

### connectivity
from joblib import Parallel, delayed
Parallel(n_jobs=10, verbose=5)(delayed(fmri_connectivity)\
(func_files[i], fmasker, mmasker, subject_list[i]) for i in range(len(func_files)))
"""
for i in np.arange(1, len(func_files), 2):
    print i
    fmri_connectivity(func_files[i], fmasker, lmasker, subject_list[i])
"""

