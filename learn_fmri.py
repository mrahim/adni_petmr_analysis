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
from nilearn.plotting import plot_img
from nilearn.input_data import MultiNiftiMasker
from sklearn.cross_validation import KFold
from sklearn.svm import SVC

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


masker = MultiNiftiMasker(standardize=False, mask_strategy='epi', 
                          memory_level=2, memory=CACHE_DIR, n_jobs=10)

data = masker.fit_transform(func_files)

correlations = []
for i in np.arange(len(func_files)):
    img = nib.load(func_files[i])
    pcc_values = img.get_data()[26, 22, 28, ...]
    
    corr = []
    for j in np.arange(data[i].shape[1]):
        corr.append(np.corrcoef(data[i][:,j], pcc_values)[0,1])
    correlations.append(corr)

###
X = np.nan_to_num(np.array(correlations))

g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0


np.savez(os.path.join(FEAT_DIR, 'features_voxels_corr_fmri'), x=x, y=y, masker=masker)


from sklearn.cross_validation import ShuffleSplit
svm = SVC(kernel='linear')
ss = ShuffleSplit(n=len(y), n_iter=100)
scores = []
for train, test in ss:
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    svm.fit(x_train, y_train)
    scores.append(svm.score(x_test, y_test))
    
import matplotlib.pyplot as plt
plt.boxplot(scores)

