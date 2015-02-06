# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 18:41:34 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_petmr, fetch_adni_masks,\
                       set_features_base_dir, set_cache_base_dir
import matplotlib.pyplot as plt

from nilearn.decoding import SpaceNetRegressor


def array_to_niis(data, mask_img):
    data_ = np.zeros(data.shape[:1] + mask_img.shape)
    data_[:, mask_img.get_data().astype(np.bool)] = data[..., ..., 0]
    data_ = np.transpose(data_, axes=(1,2,3,0))
    return nib.Nifti1Image(data_, mask_img.get_affine())

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



### Set paths
FEAT_DIR = set_features_base_dir()
CACHE_DIR = set_cache_base_dir()
CORR_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')
FIG_PATH = os.path.join(CACHE_DIR, 'figures')


### Load masks and dataset
masks = fetch_adni_masks()

dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])

idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

### load fMRI features
X = []
for i in np.arange(len(fmri)):
    X.append(np.load(os.path.join(CORR_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)


mask_img = nib.load(masks['mask_petmr'])
img = array_to_niis(X, mask_img)


y = np.zeros(X.shape[0])
y[idx['AD']]=1


snr = SpaceNetRegressor(penalty='smooth-lasso', eps=1e-1,
                        memory=CACHE_DIR, n_jobs=10)
snr.fit(img, y)




"""
groups = [['AD' , 'Normal']]
scores = []
coeffs = []
for k in range(X.shape[2]):
    X = np.array(img[k])
    score = []
    for gr in groups:
        g1_feat = X[idx[gr[0]][0]]
        idx_ = idx[gr[1]][0]
        g2_feat = X[idx_]
        x = np.concatenate((g1_feat, g2_feat), axis=0)
        y = np.ones(len(x))
        y[len(x) - len(g2_feat):] = 0

               
        sss = StratifiedShuffleSplit(y, n_iter=5, test_size=.2)
        cpt = 0
        score = []
        coeff = []
        for train, test in sss:
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            
            decoder = SpaceNetClassifier(penalty='smooth-lasso', n_jobs=10, memory=CACHE_DIR)
            decoder.fit(x_train, y_train)
            score.append(decoder.score(x_test,y_test))
            coeff.append(decoder.all_coef_)
            cpt += 1
            print cpt
    scores.append(score)
    coeffs.append(np.mean(coeff, axis=0))
    break
"""