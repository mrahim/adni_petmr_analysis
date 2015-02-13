# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:30:02 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from fetch_data import fetch_adni_petmr, fetch_adni_masks,\
                       set_cache_base_dir, set_features_base_dir


def get_sphere_coords(center, radius):
    coords_list = [np.array(center)]

    for r in np.linspace(0, radius, 10):
        for theta in np.linspace(0, np.pi, 10):
            for phi in np.linspace(0, 2*np.pi, 10):
                x = center[0] + r*np.sin(theta)*np.cos(phi)
                y = center[1] + r*np.sin(theta)*np.sin(phi)
                z = center[2] + r*np.cos(theta)
                coords_list.append(np.array([x, y, z]))
                
    return np.vstack(coords_list)

def mni_to_indices(mni_coords_list, affine):
    
    ### inverse affine
    inv_affine = np.linalg.inv(affine)
    ### transform    
    mni_coords_list = np.hstack((mni_coords_list,
                                 np.ones((mni_coords_list.shape[0],1))))
    coords_list = np.tensordot(inv_affine.reshape(4, 4, 1),
                               mni_coords_list.T.reshape(4, 1, mni_coords_list.shape[0])).T[:,:-1]
                           
    ### extract unique coords
    coords_list = np.floor(coords_list)
    cl = np.ascontiguousarray(coords_list).view(np.dtype((np.void, coords_list.dtype.itemsize * coords_list.shape[1])))
    _, idx = np.unique(cl, return_index=True)
    coords_list_unique = np.array(coords_list[idx], dtype=np.int)
    return coords_list_unique



def create_seeds_image(seeds_mni_coords, ref_img):
    img_affine = nib.load(ref_img).get_affine()
    img_shape = nib.load(ref_img).get_shape()
    seeds_img = None
    seeds_data = np.zeros(img_shape)
    print img_shape
    
    for i in range(len(seeds_mni_coords)):
        mni_coords_list = get_sphere_coords(seeds_mni_coords[i], 6)
        seed_indices = mni_to_indices(mni_coords_list, img_affine)
    
        idx_ = tuple((seed_indices[:,0].T, seed_indices[:,1].T, seed_indices[:,2].T))
        seeds_data[idx_] = i+1
    
    seeds_img = nib.Nifti1Image(seeds_data, img_affine)
        
    return seeds_img


### set paths
FEAT_DIR = set_features_base_dir()
CACHE_DIR = set_cache_base_dir()
FIG_PATH = os.path.join(CACHE_DIR, 'figures')
FMRI_PATH = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')

### fetch masks
mask = fetch_adni_masks()

### fetch fmri
dataset = fetch_adni_petmr()
func_files = dataset['func']


seeds_mni_coords=[[0, -52, 30], #PCC
                  [0, -90, 4], #Visual
                  [-34, -24, 60], #Motor-L
                  [0, 26, 28], #ACC
                  [-38, -46, 42], #LPC
                  [44, -46, 48], #RPC
                  [0, 54, 18]] #Medial prefrontal

from nilearn.datasets import fetch_msdl_atlas
atlas = fetch_msdl_atlas()
seeds_mni_coords = np.loadtxt(atlas['labels'], dtype=np.float, delimiter=',\t',
                          skiprows=1, usecols=(0,1,2))


seeds_img = create_seeds_image(seeds_mni_coords, mask['mask_pet'])

seeds_img.to_filename(os.path.join(FEAT_DIR, 'masks', 'msdl_seeds_fmri.nii.gz'))

### Test on image
lmasker = NiftiLabelsMasker(labels_img=seeds_img, detrend=True, standardize=True)
lmasker.labels_img_ = seeds_img
seeds_values = lmasker.transform(func_files[0])
