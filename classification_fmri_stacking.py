# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 11:26:09 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import set_cache_base_dir, fetch_adni_petmr
from fetch_data import set_features_base_dir
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')
CACHE_DIR = set_cache_base_dir()
FIG_PATH = os.path.join(CACHE_DIR, 'figures')


dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])

idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)


### load fMRI features
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(FMRI_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)

g1_feat = X[idx['AD'][0]]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)    
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

sss = StratifiedShuffleSplit(y, n_iter=10, test_size=.2)
scores = []
pr_score = []
train_predictions = []
test_predictions = []

for train, test in sss:
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    for k in range(x.shape[2]):
        print k
        svm = SVC(kernel='linear', probability=True)
        svm.fit(x_train[..., k], y_train)
        scores.append(svm.score(x_test[..., k], y_test))
        train_predictions.append(svm.predict_proba(x_train[..., k]))
        test_predictions.append(svm.predict_proba(x_test[..., k]))
    
    pr_train = np.asarray(train_predictions)[..., 0].T
    pr_test = np.asarray(test_predictions)[..., 0].T
    stc = SVC(kernel='linear')
    stc.fit(pr_train, y_train)
    pr_score.append(stc.score(pr_test, y_test))


sc = []
for i in range(7):
    sc.append(scores[:(i*10)+9])
    
plt.figure()
plt.boxplot(sc)

plt.figure()
plt.boxplot(pr_score)


"""
scores = []
predictions = []
sss = StratifiedShuffleSplit(y, n_iter=10, test_size=.2)

for k in range(X.shape[2]):


    cpt = 0
    score = []
    coeff = []
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        score.append(svm.score(x_test, y_test))
        predictions.append(svm.predict(x_train))
        cpt += 1
        print k, cpt
    scores.append(score)
    
"""