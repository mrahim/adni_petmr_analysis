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
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit

### set paths
FIG_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp', 'figures', 'petmr')
FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
corr_path = os.path.join(FEAT_DIR, 'features_fmri_masked.npz')
pet_model_path = os.path.join(FEAT_DIR, 'coef_map_pet.npz')

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

### Ridge with variable substitution
lambda_ = .7
w_pet = np.array(model)
Y = y - lambda_ * np.dot(x, w_pet.T)
X = x

rdg = RidgeCV(alphas=np.logspace(-3, 3, 70))
rdg.fit(X,Y)

### original weights
w = rdg.coef_[0, :].T - lambda_ * w_pet
y_predict = np.dot(X, w.T)

### logistic regression on ridge
lgr = LogisticRegression()
lgr.fit(y_predict, y)
print lgr.score(y_predict, y)

### fmri classification
lgr = LogisticRegression()
lgr.fit(x, y)
print lgr.score(x, y)

### ShuffleSplit comparison
ss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)
fmri_scores = []
rdg_scores = []
cpt = 0
for train, test in ss:
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]
    lgr = LogisticRegression()
    lgr.fit(x_train, y_train)
    s1 = lgr.score(x_test, y_test)
    fmri_scores.append(s1)
    
    y_predict = np.dot(x_test, w.T)
    lgr = LogisticRegression()
    lgr.fit(y_predict, y_test)
    s2 = lgr.score(y_predict, y_test)
    rdg_scores.append(s2)
    cpt += 1
    print cpt, s1, s2
    

bp = plt.boxplot([fmri_scores, rdg_scores])
for key in bp.keys():
    for box in bp[key]:
        box.set(linewidth=2)
plt.grid(axis='y')
plt.xticks(range(1,3), ['fMRI', 'fMRI+PET model'], fontsize=17)
plt.ylabel('Accuracy (%)', fontsize=17)
plt.ylim([0.2, 1.0])
plt.yticks(np.linspace(.2, 1.0, 9), np.arange(20,110,10), fontsize=17)
plt.title('AD/Normal classification accuracies', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'output1.png'))
plt.savefig(os.path.join(FIG_DIR, 'output1.pdf'))
