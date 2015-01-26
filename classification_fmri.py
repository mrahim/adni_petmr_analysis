# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

@author: mehdi.rahim@cea.fr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:16:46 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_petmr, fetch_adni_fdg_pet
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt

FIG_PATH = '/disk4t/mehdi/data/tmp/figures'

def plot_shufflesplit(score, groups, title, filename):
    bp = plt.boxplot(score, 0, '', 0)
    for key in bp.keys():
        for box in bp[key]:
            box.set(linewidth=2)
    plt.grid(axis='x')
    plt.xlim([.4, 1.])
    plt.xlabel('Accuracy (%)', fontsize=18)
    if len(title)==0:
        title = 'Shuffle Split Accuracies '
    plt.title(title, fontsize=17)
    ylabels = []
    for g in groups:
        ylabels.append('/'.join(g))
    plt.yticks(range(1,7), ylabels, fontsize=16)
    plt.xticks(np.linspace(0.4,1.0,7), np.arange(40,110,10), fontsize=18)
    plt.tight_layout()
    if len(filename)==0:
        filename = 'boxplot_adni_baseline_petmr'
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join([filename,
                          ext])
        plt.savefig(os.path.join(FIG_PATH, fname), transparent=False)


FEAT_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features')
CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'tmp')
CORR_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'features',
                        'fmri_subjects', 'corr_subjects')


dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])

idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

### load features
X = []
for i in np.arange(len(fmri)):
    X.append(np.load(os.path.join(CORR_DIR, subj_list[i]+'.npy')))
X = np.array(X)

### Classification
#groups = [['Normal'], ['EMCI'], ['LMCI'], ['EMCI', 'Normal'], ['LMCI', 'EMCI', 'Normal']]
groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'], 
          ['LMCI', 'Normal'], ['LMCI', 'EMCI'], ['EMCI', 'Normal']]
scores = []
for gr in groups:
    g1_feat = X[idx[gr[0]][0]]
    idx_ = idx[gr[1]][0]
    g2_feat = X[idx_]
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0

    sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)
    
    cpt = 0
    score = []
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        score.append(svm.score(x_test, y_test))
        cpt += 1
        print cpt
    scores.append(score)
"""
for gr in groups:
    g1_feat = X[idx['AD'][0]]
    idx_ = idx[gr[0]][0]
    for k in np.arange(1, len(gr)):
        idx_ = np.hstack((idx_, idx[gr[k]][0]))
    g2_feat = X[idx_]
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0

    sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)
    
    cpt = 0
    score = []
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
        svm = SVC(kernel='linear')
        svm.fit(x_train, y_train)
        score.append(svm.score(x_test, y_test))
        cpt += 1
        print cpt
    scores.append(score)
"""