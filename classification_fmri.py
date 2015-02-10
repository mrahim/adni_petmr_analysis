# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import set_cache_base_dir, fetch_adni_petmr
from fetch_data import set_features_base_dir
import matplotlib.pyplot as plt

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


groups = [['AD' , 'Normal']]
scores = []
coeffs = []
for k in range(X.shape[2]):
    score = []
    for gr in groups:
        g1_feat = X[idx[gr[0]][0]]
        idx_ = idx[gr[1]][0]
        idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
        g2_feat = X[idx_]
        x = np.concatenate((g1_feat[...,k], g2_feat[...,k]), axis=0)
        y = np.ones(len(x))
        y[len(x) - len(g2_feat):] = 0
        
        sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2)
        
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
            coeff.append(svm.coef_)
            cpt += 1
            print k, cpt
    scores.append(score)
    coeffs.append(np.mean(coeff, axis=0))

"""
### SVM coeffs
for k in range(X.shape[2]):
    np.savez_compressed(os.path.join(FEAT_DIR, 'fmri_models',
                                     'svm_coeffs_fmri_seed_'+str(k)),
                        svm_coeffs=coeffs[k],
                        idx=idx,
                        subjects=dataset['subjects'])
"""