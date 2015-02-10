# -*- coding: utf-8 -*-
"""
Script that computes svm coeffs on PET voxels
Created on Mon Jan 26 11:16:46 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_petmr, fetch_adni_fdg_pet,\
                       fetch_adni_fdg_pet_diff, fetch_adni_masks
import matplotlib.pyplot as plt

FIG_PATH = '/disk4t/mehdi/data/tmp/figures'

def plot_shufflesplit(score, groups, title):
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
        ylabels.append(','.join(g))
    plt.yticks(range(1,7), ylabels, fontsize=16)
    plt.xticks(np.linspace(0.4,1.0,7), np.arange(40,110,10), fontsize=18)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join(['boxplot_adni_baseline_petmr',
                          ext])
        plt.savefig(os.path.join(FIG_PATH, fname), transparent=False)

from nilearn.masking import apply_mask

###
def learn_pet_coeffs(img_files, indices, mask):
    """Returns SVM coeffs learned on PET dataset
    """
    
    n_iter = 100    
    
    X = apply_mask(img_files, mask)
    
    ### AD / NC classification
    g1_feat = X[idx['AD'][0]]
    idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
    #g2_feat = X[idx['Normal'][0]]
    g2_feat = X[idx_]
    
    
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0
    sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2)
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
        coeff.append(svm.coef_)
        score.append(svm.score(x_test, y_test))
        cpt += 1
        print cpt,'/',str(n_iter)
    
    return np.mean(coeff, axis=0), np.array(score)
    
from fetch_data import set_cache_base_dir, set_features_base_dir

FEAT_DIR = set_features_base_dir() 
PET_DIR = os.path.join(FEAT_DIR, 'pet_models')
CACHE_DIR = set_cache_base_dir()

### Load pet data and mask
mask = fetch_adni_masks()

datasets = {}
#datasets['petmr'] = fetch_adni_petmr()
#datasets['pet_diff']  = fetch_adni_fdg_pet_diff()
datasets['pet'] = fetch_adni_fdg_pet()
all_scores = []

for key in datasets.keys():
    print key
    dataset = datasets[key]
    dx_group = np.array(dataset['dx_group'])
    idx = {}
    for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
        idx[g] = np.where(dx_group == g)
    coeffs, scores = learn_pet_coeffs(dataset['pet'], idx, mask['mask_petmr'])
    np.savez_compressed(os.path.join(PET_DIR, 'ad_mci_svm_coeffs_' + key),
                        svm_coeffs=coeffs, subjects=dataset['subjects'],
                        idx=idx)
    all_scores.append(scores)