# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:52:06 2015

@author: mehdi.rahim@cea.fr
"""


import os
from fetch_data import fetch_adni_petmr
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_roi
from nilearn.masking import intersect_masks
from nilearn.image import resample_img, index_img

CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
MASK_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features', 'masks')

dataset = fetch_adni_petmr()
func_files = dataset['func']
pet_files = dataset['pet']


# Create PET mask
pet_masker = MultiNiftiMasker(mask_args=dict(opening=1, lower_cutoff=0.5, upper_cutoff=0.6), mask_strategy='epi', n_jobs=1, memory=CACHE_DIR, memory_level=2)
pet_masker.fit(pet_files)
pet_masker.mask_img_.to_filename(os.path.join(MASK_DIR, 'mask_pet.nii.gz'))


# Create fMRI mask
fmri_masker = MultiNiftiMasker(mask_strategy='epi', n_jobs=1, memory=CACHE_DIR, memory_level=2)
fmri_masker.fit(func_files)
fmri_masker.mask_img_.to_filename(os.path.join(MASK_DIR, 'mask_fmri.nii.gz'))


# Intersect the masks
pet_mask = pet_masker.mask_img_
fmri_mask = fmri_masker.mask_img_

# 1- Resample fMRI to PET resolution
fmri_resampled_mask = resample_img(fmri_mask,
                                  target_affine=pet_mask.get_affine(),
                                  target_shape=pet_mask.get_shape(),
                                  interpolation='nearest')

# 2- Intersect the masks                                  
petmr_mask = intersect_masks([pet_mask, fmri_resampled_mask])
petmr_mask.to_filename(os.path.join(MASK_DIR, 'mask_petmr.nii.gz'))


# Plot the masks !
plot_roi(petmr_mask, bg_img=index_img(func_files[0],0),
         title='Intersection mask on fMRI')
plot_roi(petmr_mask, bg_img=pet_files[0],
         title='Intersection mask on PET')
plot_roi(pet_mask, bg_img=pet_files[0], title='PET mask')
plot_roi(fmri_mask, bg_img=index_img(func_files[0],0), title='fMRI mask')