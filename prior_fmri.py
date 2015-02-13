# -*- coding: utf-8 -*-
"""
Use the 


Created on Wed Feb 11 15:54:46 2015

@author: mehdi.rahim@cea.fr
"""

from priorclassifier import PriorClassifier
import os
import numpy as np
from fetch_data import set_features_base_dir, fetch_adni_petmr,\
                        set_cache_base_dir

from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import RidgeCV, LogisticRegression
import matplotlib.pyplot as plt

### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects')


### load fMRI features
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

### load PET a priori
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)
w_pet = w_pet/np.max(w_pet)
### prepare data
g1_feat = X[idx['AD'][0]]
#idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
x = np.concatenate((g1_feat, g2_feat), axis=0)
y = np.ones(len(x))
y[len(x) - len(g2_feat):] = 0

n_iter = 10

sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))
ss = ShuffleSplit(len(y), n_iter=n_iter, test_size=.2,
                  random_state=np.random.seed(42))

rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
regressor = {'ridge': rdgc}

all_scores = {}

for key in regressor.keys():
    print key
    scores = []
    scores_prior = []
    
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        
        sc = []
        x_train_stacked_prior = []
        x_test_stacked_prior = []
        x_train_stacked = []
        x_test_stacked = []
        for k in range(7):
            xtrain = x_train[..., k]
            xtest = x_test[..., k]
            
            rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
            rdgc.fit(xtrain, y_train)
            x_train_stacked.append(rdgc.predict(xtrain))
            x_test_stacked.append(rdgc.predict(xtest))
            #print rdgc.score(xtest, y_test)
                        
            rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
            pc = PriorClassifier(rdgc, w_pet, .7)
            pc.fit(xtrain, y_train)
            x_train_stacked_prior.append(pc.predict(xtrain))
            x_test_stacked_prior.append(pc.predict(xtest))
            sc.append(pc.score(xtest, y_test))
            #print 'prior', pc.score(xtest, y_test)

        x_train_ = np.asarray(x_train_stacked).T
        x_test_ = np.asarray(x_test_stacked).T
        lgr = LogisticRegression()
        lgr.fit(x_train_, y_train)
        scores.append(lgr.score(x_test_,  y_test))
        print 'stacking', lgr.score(x_test_,  y_test)

        x_train_prior_ = np.asarray(x_train_stacked_prior)[..., 0].T
        x_test_prior_ = np.asarray(x_test_stacked_prior)[..., 0].T
        lgr = LogisticRegression()
        lgr.fit(x_train_prior_, y_train)
        scores_prior.append(lgr.score(x_test_prior_,  y_test))        
        print 'stacking prior', lgr.score(x_test_prior_,  y_test)

plt.figure()
plt.boxplot([scores, scores_prior])
plt.plot([1,2],[scores, scores_prior],'--c')
plt.ylim([0.2, 1.0])
scores_prior = np.array(scores_prior)
scores = np.array(scores)
neg_idx = np.where(scores_prior - scores < 0)
plt.plot([1,2],[scores[neg_idx], scores_prior[neg_idx]],'--r')
