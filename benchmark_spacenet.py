# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:26:50 2015

@author: mehdi.rahim@cea.fr
"""

import os, sys

## Force stdout instant flush
#############################
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)


import numpy as np
from fetch_data import fetch_adni_petmr, array_to_nii, set_features_base_dir \
                       set_group_indices, array_to_niis, fetch_adni_masks
from nilearn.decoding import SpaceNetClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.datasets.base import Bunch


def train_and_test(X, y, mask, train, test,
                   penalty, n_alphas, l1_ratio, tol, max_iter):
    """Returns a bunch containing the score,
    the proba and the coeff of the train and test iteration.
    """
    x_train, y_train = X[train], y[train]
    x_test, y_test = X[test], y[test]

    spnc = SpaceNetClassifier(mask=mask,
                              penalty=penalty,
                              n_alphas=n_alphas,
                              l1_ratios=l1_ratio,
                              tol=tol, max_iter=max_iter,
                              cv=8, verbose=0)
    spnc.fit(x_train, y_train)
    model_params = spnc.best_model_params_
    ypredict = spnc.predict(x_test)
    score = accuracy_score(y_test, ypredict)
    coeff = spnc.coef_
    B = Bunch(score=score, model_params=model_params, coeff=coeff)
    np.savez_compressed(os.path.join(FEAT_DIR, 'tvl1_bench'), data=p,
                        l1_ratio=np.linspace(.1, .9, 9), penalty=penalty)
    return B


#############################################################################

mask = fetch_adni_masks()
dataset = fetch_adni_petmr()
idx = set_group_indices(dataset['dx_group'])
img_files = dataset['pet'][np.hstack((idx['AD'], idx['Normal']))]
y = np.array([1] * len(idx['AD']) + [-1] * len(idx['Normal']))

n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter, test_size=.2, random_state=42)

from joblib import Parallel, delayed
penalty = 'tv-l1'
for l1_ratio in np.linspace(.1, .9, 9):
    print 'classification with l1_ratio : ', l1_ratio
    p = Parallel(n_jobs=-1, verbose=5)\
                (delayed(train_and_test)(img_files, y, mask['mask_petmr'],
                                         penalty=penalty, n_alphas=10,
                                         l1_ratios=l1_ratio, tol=1e-6,
                                         max_iter=10000)\
                for train, test in sss)
    np.savez_compressed(os.path.join(FEAT_DIR, 'tvl1_bench'), data=p,
                        l1_ratio=l1_ratio, penalty=penalty)

