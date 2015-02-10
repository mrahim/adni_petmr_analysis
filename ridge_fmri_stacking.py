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
from fetch_data import set_cache_base_dir, set_features_base_dir




def ridge_apriori(a, y, w_pet, lambda_=.7, k=0, n_iter=100):
    x = a[...,k]    
    Y = y - lambda_ * np.dot(x, w_pet.T)
    X = x
    
    rdg = RidgeCV(alphas=np.logspace(-3, 3, 70))
    rdg.fit(X,Y)
    
    ### original weights
    w = rdg.coef_[0, :].T - lambda_ * w_pet
    y_predict = np.dot(X, w.T)
    
    ### logistic regression on ridge + a priori
    lgr = LogisticRegression()
    lgr.fit(y_predict, y)
    
    ### fmri classification
    lgr = LogisticRegression()
    lgr.fit(x, y)
    
    ### ShuffleSplit comparison
    ss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2)
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
        print cpt
    return 0


### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')


### load fMRI features
#corr_path = os.path.join(FEAT_DIR, 'features_fmri_masked.npz')
dataset = fetch_adni_petmr()
fmri = dataset['func']
subj_list = dataset['subjects']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)
X = []
for i in np.arange(len(subj_list)):
    X.append(np.load(os.path.join(FMRI_DIR, subj_list[i]+'.npz'))['corr'])
X = np.array(X)


### load pet model
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'ad_mci_svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']

### prepare data
g1_feat = X[idx['AD'][0]]
idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

a = np.copy(x)
### Ridge with variable substitution
for k in range(7):
    
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
    ss = StratifiedShuffleSplit(y, n_iter=50, test_size=.2)
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
    plt.title('AD/MCI classification accuracies', fontsize=18)
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