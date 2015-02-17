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



def train_and_test(train, test, g1_feat, g2_feat):
    x_train_stacked_prior = []
    x_test_stacked_prior = []
    x_train_stacked = []
    x_test_stacked = []
    for k in range(g1_feat.shape[2]):
        x = np.concatenate((g1_feat[..., k], g2_feat[..., k]), axis=0)
        xtrain = x[train]
        y_train = y[train]
        xtest = x[test]
        y_test = y[test]

        """
        xtrain = x_train[..., k]
        xtest = x_test[..., k]
        """
        
        rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
        rdgc.fit(xtrain, y_train)
        x_train_stacked.append(rdgc.predict(xtrain))
        x_test_stacked.append(rdgc.predict(xtest))
        #print rdgc.score(xtest, y_test)
                    
        rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
        pc = PriorClassifier(rdgc, w_pet, .7)
        #pc.fit(xtrain, y_train)
        pc.fit(x[:,...], y[:])
        x_train_stacked_prior.append(pc.predict(xtrain))
        x_test_stacked_prior.append(pc.predict(xtest))
        #sc.append(pc.score(xtest, y_test))
        #print 'prior', pc.score(xtest, y_test)

    x_train_ = np.asarray(x_train_stacked).T
    x_test_ = np.asarray(x_test_stacked).T
    lgr = LogisticRegression()
    lgr.fit(x_train_, y_train)
    #scores.append(lgr.score(x_test_,  y_test))
    a = lgr.score(x_test_,  y_test)

    x_train_prior_ = np.asarray(x_train_stacked_prior)[..., 0].T
    x_test_prior_ = np.asarray(x_test_stacked_prior)[..., 0].T
    lgr = LogisticRegression()
    lgr.fit(x_train_prior_, y_train)
    #scores_prior.append(lgr.score(x_test_prior_,  y_test))
    b = lgr.score(x_test_prior_,  y_test)
    return [a, b]



### set paths
CACHE_DIR = set_cache_base_dir()
FIG_DIR = os.path.join(CACHE_DIR, 'figures', 'petmr')
FEAT_DIR = set_features_base_dir()
FMRI_DIR = os.path.join(FEAT_DIR, 'smooth_preproc', 'fmri_subjects_68rois')


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
X = np.array(X, copy=False)

### load PET a priori
pet_model_path = os.path.join(FEAT_DIR, 'pet_models',
                              'ad_mci_svm_coeffs_pet_diff.npz')
model = np.load(pet_model_path)['svm_coeffs']
w_pet = np.array(model)
#w_pet = w_pet/np.max(w_pet)
### prepare data
g1_feat = X[idx['AD'][0]]
#idx_ = idx['Normal'][0]
idx_ = np.hstack((idx['LMCI'][0], idx['EMCI'][0]))
g2_feat = X[idx_]
y = np.ones(len(idx['AD'][0]) + len(idx_))
y[len(y) - len(g2_feat):] = 0

n_iter = 100

sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.2,
                             random_state=np.random.seed(42))
ss = ShuffleSplit(len(y), n_iter=n_iter, test_size=.2,
                  random_state=np.random.seed(42))

rdgc = RidgeCV(alphas=np.logspace(-3, 3, 7))
regressor = {'ridge': rdgc}

all_scores = {}
from joblib import Parallel, delayed


p = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)\
(train, test, g1_feat, g2_feat) for train, test in sss)
    
"""
plt.figure()
plt.boxplot([scores, scores_prior])
plt.plot([1,2],[scores, scores_prior],'--c')
plt.ylim([0.2, 1.0])
scores_prior = np.array(scores_prior)
scores = np.array(scores)
neg_idx = np.where(scores_prior - scores < 0)
plt.plot([1,2],[scores[neg_idx], scores_prior[neg_idx]],'--r')
"""

#np.savez_compressed('prior_msdl_atlas', scores=scores, scores_prior=scores_prior)

