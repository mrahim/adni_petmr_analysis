# -*- coding: utf-8 -*-
"""

Estimate lambda : a scaling parameter between w and w_pet
1- Load w_pet (PET svm weights)
2- Estimate w_univ (Least square regression)
3- Estimate lambda with : min(lambda)(|lambda w_pet - w_univ|)

Created on Tue Jan 27 10:33:32 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import matplotlib.pyplot as plt


FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
pet_model_path = os.path.join(FEAT_DIR, 'coef_map_pet.npz')
#pet_model_path = os.path.join(FEAT_DIR, 'coef_mean_map_pet.npz')
corr_path = os.path.join(FEAT_DIR, 'features_fmri_masked.npz')

### load pet model
npz = np.load(pet_model_path)
model = npz['coef_map']

### load fmri features
npz = np.load(corr_path)
X = npz['correlations']
idx = npz['idx'].any()

### prepare data
g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0
from sklearn.preprocessing import scale
x = scale(x)

### estimate w_univ
b = np.dot(x.T, y)
dg = np.zeros(x.shape[1])
for i in np.arange(x.shape[1]):
    dg[i] = np.inner(x[:,i], x[:,i])
w_univ = b/dg

### estimate alpha
m = np.hstack(model)
lambda_ = np.inner(m, w_univ)/np.sum(np.power(m,2))
print lambda_