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
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
FMRI_PATH = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features', 'fmri_subjects')

### fetch fmri
dataset = fetch_adni_petmr()
func_files = dataset['func']
subject_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

pcc_mni_coords = [0, -44, 34]

img = nib.load(func_files[0])
affine = img.get_affine()
mni_coords_list = get_sphere_coords(pcc_mni_coords, 3.5)
pcc_indices = mni_to_indices(mni_coords_list, affine)

seed_values = img.get_data()[tuple((pcc_indices[:,0].T,pcc_indices[:,1].T,pcc_indices[:,2].T))]

d = np.load(os.path.join(FMRI_PATH, subject_list[0]+'.npy'))