# -*- coding: utf-8 -*-
"""
Compute a ridge that combines PET model and fMRI correlations.
The general formula is :
|Xw - y|^2 + alpha |w - lambda w_tep|^2

By making :
beta = w - lambda w_tep

We have :
|X beta - (y - lambda X w_tep)|^2 + alpha |beta|^2


Created on Wed Jan 21 09:05:28 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from sklearn.linear_model import RidgeCV

### set paths
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
corr_path = os.path.join(FEAT_DIR, 'features_fmri_masked.npz')
pet_model_path = os.path.join(FEAT_DIR, 'coef_map_pet.npz')

### load pet model
npz = np.load(pet_model_path)
idx = npz['idx'].any()
model = npz['coef_map']

### load fmri features
npz = np.load(corr_path)
X = npz['correlations']

### prepare data
g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

### Ridge with variable substitution
lambda_ = .7
w_pet = np.array(model)
Y = y - lambda_ * np.dot(x, w_pet.T)
X = x

rdg = RidgeCV(alphas=np.logspace(-3, 3, 70))
rdg.fit(X,Y)

### original weights
w = rdg.coef_ - lambda_ * w_pet
y_predict = np.dot(X, w)
print y_predict