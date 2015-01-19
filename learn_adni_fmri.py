# -*- coding: utf-8 -*-
"""

Learn ADNI fMRI

Created on Mon Jan 19 14:17:03 2015

@author: mr243268
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr
from nilearn.datasets import fetch_msdl_atlas
from nilearn.input_data import NiftiMapsMasker
import matplotlib.pyplot as plt


FIG_PATH = '/disk4t/mehdi/data/tmp/figures'
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
                         
dataset = fetch_adni_petmr()

func_files = dataset['func']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

n_subjects = len(func_files)
subjects = []
corr_feat = []
corr_mat = []
PCC_COORDS = [26, 22, 28] #[0, -44, 34]


for f in func_files:
    img = nib.load(f)
    pcc_values = img.get_data()[26, 22, 28, ...]
    np.correlate(pcc_values, )
    break