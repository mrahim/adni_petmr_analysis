# -*- coding: utf-8 -*-
"""
Compute PCC correlation features at voxel level on rs-fmri

Created on Tue Jan 20 13:15:00 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr, fetch_adni_masks
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.image import index_img
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


def fmri_connectivity(func_file, img_masker, seeds_masker, subject_id):
    ### 1-Extract seeds
    ### 2-Extract fMRI voxels
    ### 3-Compute correlations for each voxel

    if not os.path.isfile(os.path.join(FMRI_DIR, subject_id + '.npz')):
        
        seeds_values = seeds_masker.transform(func_file)
        fmri_values = img_masker.transform(func_file)
        
        n_seeds = seeds_values.shape[1]
        n_voxels = fmri_values.shape[1]
        print seeds_values.shape, fmri_values.shape
        c = []
        for i in range(n_voxels):
            fmri_vox = np.tile(fmri_values[:, i], (39, 1)).T
            c.append(fast_corr(seeds_values, fmri_vox))
        np.savez_compressed(os.path.join(FMRI_DIR, subject_id),
                            corr=np.array(c))

"""        
        seeds_values = np.tile(np.tile(seeds_values, (1,1,1)), (n_voxels,1,1))
        fmri_values = np.tile(np.tile(fmri_values, (1,1,1)), (n_seeds,1,1))
    
        seeds_values = np.transpose(seeds_values, (1,0,2))
        fmri_values = np.transpose(fmri_values, (1,2,0))

        print fmri_values.shape
        print seeds_values.shape            
        c = fast_corr(seeds_values, fmri_values)
        np.savez_compressed(os.path.join(FMRI_DIR, subject_id), corr=c)
"""
#######################################################
#######################################################

### set paths
from fetch_data import set_cache_base_dir, set_features_base_dir
CACHE_DIR = set_cache_base_dir()
FIG_PATH = os.path.join(CACHE_DIR, 'figures')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_msdl')

### fetch fmri, load masks and seeds
dataset = fetch_adni_petmr()
func_files = dataset['func']
subject_list = dataset['subjects']
mask = fetch_adni_masks()
seeds_img = os.path.join(FEAT_DIR, 'masks', 'msdl_seeds_fmri.nii.gz')

### Labels
lmasker = NiftiLabelsMasker(labels_img=seeds_img, mask_img=mask['mask_petmr'],
                            resampling_target='labels', detrend=True,
                            standardize=False, t_r=3.,
                            memory=CACHE_DIR, memory_level=2)
lmasker.labels_img_ = nib.load(seeds_img)
lmasker.mask_img_ = nib.load(mask['mask_petmr'])

### fMRI
fmasker = NiftiMasker(mask_img=mask['mask_petmr'], detrend=False, t_r=3.,
                      memory=CACHE_DIR, memory_level=2)
fmasker.mask_img_ = nib.load(mask['mask_petmr'])

### connectiviy
for i in range(1):
    fmri_connectivity(func_files[i], fmasker, lmasker, subject_list[i])
