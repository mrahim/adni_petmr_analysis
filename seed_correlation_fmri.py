# -*- coding: utf-8 -*-
"""
funcitions for the:
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
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')

### fetch fmri
dataset = fetch_adni_petmr()
func_files = dataset['func']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

### 


img = nib.load(func_files[0])