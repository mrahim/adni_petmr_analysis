# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:47:54 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from fetch_data import set_group_indices, fetch_adni_rs_fmri_conn,\
                        set_cache_base_dir, set_features_base_dir,\
                        fetch_adni_rs_fmri
                        
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.datasets import fetch_msdl_atlas
import matplotlib.pyplot as plt
from sklearn.covariance import GraphLassoCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import LinearSVC


def train_and_test(clf, X, y, train, test):
    """ One iteration of classification
    """
    
    clf.fit(X[train, :], y[train])
    return clf.score(X[test, :], y[test])


def extract_signal(masker, dataset, k):
    """Returns signal matrix (times x regions)
    """
    s = masker.fit_transform(dataset.func[k])
    gl = GraphLassoCV()
    gl.fit(s)
    ind = np.tril_indices(39, k=-1)
    c = gl.covariance_[ind]
    print c.shape
    return c

###################################
dataset = fetch_adni_rs_fmri()
atlas = fetch_msdl_atlas()['maps']

print dataset.func[0]

masker = NiftiMapsMasker(maps_img=atlas, detrend=True,
                         smoothing_fwhm=5, verbose=5)
                         
from joblib import delayed, Parallel

p = Parallel(n_jobs=20, verbose=5)(delayed(extract_signal)\
                                          (masker, dataset, k)\
                                          for k in range(len(dataset.func)))

x = np.array(p)

idx = set_group_indices(dataset['dx_group'])
idx_ = np.hstack((idx['EMCI'], idx['LMCI']))
img_idx = np.hstack((idx['AD'], idx_))

x = x[img_idx, :]
y = np.ones(x.shape[0])
y[len(y) - len(idx_):] = 0

svc = LinearSVC(penalty='l1', dual=False)
sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.25, random_state=42)
s = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)\
                                          (svc, x, y, train, test)\
                                          for train, test in sss):



