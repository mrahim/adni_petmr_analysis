# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:37:27 2015

@author: mehdi.rahim@cea.fr
"""

import os, sys

## Force stdout instant flush
#############################
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)


import numpy as np
from fetch_data import fetch_adni_petmr, array_to_nii, \
                       set_group_indices, array_to_niis, fetch_adni_masks
from nilearn.decoding import SpaceNetClassifier, SpaceNetRegressor
from nilearn.masking import apply_mask
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from nilearn.plotting import plot_stat_map

mask = fetch_adni_masks()
dataset = fetch_adni_petmr()
idx = set_group_indices(dataset['dx_group'])
img_files = dataset['pet'][np.hstack((idx['AD'], idx['Normal']))]
y = np.array([1] * len(idx['AD']) + [-1] * len(idx['Normal']))

sss = StratifiedShuffleSplit(y, 1, test_size=.2, random_state=42)

for train, test in sss:
    x_train = img_files[train]
    y_train = y[train]
    x_test = img_files[test]
    y_test = y[test]
    
    spnc = SpaceNetClassifier(mask=mask['mask_petmr'],
                              alphas=np.logspace(-3, 3, 7),
                              l1_ratios=np.linspace(.1, .9, 9),
                              cv=8,
                              n_jobs=20)
    spnc.fit(x_train, y_train)
    print spnc.best_model_params_
    ypredict = spnc.predict(x_test)    
    print accuracy_score(y_test, ypredict)

    img = array_to_nii(spnc.coef_[0, :], mask['mask_petmr'])
    plot_stat_map(img, cut_coords=(7, -70, 34))
