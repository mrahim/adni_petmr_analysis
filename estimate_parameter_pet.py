# -*- coding: utf-8 -*-
"""
w_pet estimation by a least square method

w = (x.T * x)^-1 * x.T * y
w = [ diag(x.T * x) ]^-1 * x.T * y


Created on Wed Jan 21 18:15:49 2015

@author: mehdi.rahim@cea.fr
"""
import os
import numpy as np

### set paths
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
feat_path = os.path.join(FEAT_DIR, 'features_voxels_norm_pet.npz')
pet_model_path = os.path.join(FEAT_DIR, 'coef_map_pet.npz')

### load data
npz = np.load(feat_path)
X = npz['X']
idx = npz['idx'].all()

### prepare data
g1_feat = X[idx['AD'][0]]
g2_feat = X[idx['Normal'][0]]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

#w = np.dot(np.linalg.inv(np.diag(np.inner(x, x))), np.dot(x.T, y))

from sklearn.preprocessing import scale
x = scale(x)

import time
tic = time.clock()
b = np.dot(x.T, y)

dg = np.zeros(x.shape[1])
for i in np.arange(x.shape[1]):
    dg[i] = np.inner(x[:,i], x[:,i])
w = b/dg

toc = time.clock()
print toc-tic

npz = np.load(pet_model_path)
model = npz['coef_map']

y_lsr = np.vstack(np.dot(x,w))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(y_lsr,y)
print lr.score(y_lsr,y)


y_mod = np.vstack(np.dot(x,model.T))
lr = LogisticRegression()
lr.fit(y_mod,y)
print lr.score(y_mod,y)
