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
import matplotlib.lines as mlines
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from fetch_data import fetch_adni_petmr


### set paths

BASE_DIR = os.path.join('/', 'disk4t', 'mehdi')
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.join('/', 'home', 'mr243268')
    
FIG_DIR = os.path.join(BASE_DIR, 'data', 'tmp', 'figures', 'petmr')
FEAT_DIR = os.path.join(BASE_DIR, 'data', 'features')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'tmp')
CORR_DIR = os.path.join(BASE_DIR, 'data', 'features',
                        'smooth_preproc', 'fmri_subjects')

corr_path = os.path.join(FEAT_DIR, 'features_fmri_masked.npz')
pet_model_path = os.path.join(FEAT_DIR, 'pet_models', 'svm_coeffs_pet.npz')

###



### load petmr dataset
dataset = np.load('/home/mr243268/data/features/fmri_models/svm_coeffs_fmri_seed_0.npz')
subj_list = dataset['subjects']
idx =  np.any(dataset['idx'])
"""
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)
"""

### load pet model
model = np.load(pet_model_path)['svm_coeffs']


### load fMRI features
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(CORR_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)


### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['Normal'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

a = np.copy(x)
### Ridge with variable substitution
for k in [4]:
    
    x = a[...,k]
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
        
    
    plot_diffs = True
    rdg_scores = np.array(rdg_scores)
    fmri_scores = np.array(fmri_scores)
    neg_idx = np.where(rdg_scores - fmri_scores < 0)
    
    bp = plt.boxplot([fmri_scores, rdg_scores])
    for key in bp.keys():
        for box in bp[key]:
            box.set(linewidth=2)
    plt.grid(axis='y')
    plt.xticks([1, 2] , ['fMRI', 'fMRI+PET model'], fontsize=17)
    plt.ylabel('Accuracy (%)', fontsize=17)
    plt.ylim([0.2, 1.0])
    plt.yticks(np.linspace(.2, 1.0, 9), np.arange(20,110,10), fontsize=17)
    plt.title('AD/CN+EMCI classification accuracies', fontsize=18)
    if plot_diffs:
        plt.plot([1,2],[fmri_scores, rdg_scores],'--c')
        plt.plot([1,2],[fmri_scores[neg_idx], rdg_scores[neg_idx]],'--r')
    plt.tight_layout()
    
    blue_line = mlines.Line2D([], [], linewidth=2, color='c', label='+')
    red_line = mlines.Line2D([], [], linewidth=2, color='r', label='-')
    plt.legend(handles=[blue_line, red_line])
    plt.savefig(os.path.join(FIG_DIR, 'output'+str(k)+'.png'))
    plt.savefig(os.path.join(FIG_DIR, 'output'+str(k)+'.pdf'))
    
    
    from scipy.stats import wilcoxon
    _, pval = wilcoxon(rdg_scores, fmri_scores)
    plt.figure()
    bp = plt.boxplot(rdg_scores-fmri_scores)
    for key in bp.keys():
        for box in bp[key]:
            box.set(linewidth=2)
    plt.xticks([] , [])
    plt.ylabel('Difference', fontsize=17)
    plt.yticks(np.linspace(-.5, .5, 11), np.linspace(-.5, .5, 11), fontsize=17)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.title('Wilcoxon pval = '+str('10^-%.2f' % -np.log10(pval)), fontsize=17)
    plt.savefig(os.path.join(FIG_DIR, 'output'+str(k)+'-diff.png'))
    plt.savefig(os.path.join(FIG_DIR, 'output'+str(k)+'-diff.pdf'))