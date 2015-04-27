# -*- coding: utf-8 -*-
"""
Random forest

@author: mehdi.rahim@cea.fr
"""

import os, sys
import numpy as np
from fetch_data import set_group_indices, fetch_adni_petmr,\
                       fetch_adni_rs_fmri_conn, set_features_base_dir,\
                       set_cache_base_dir
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets.base import Bunch
from sklearn.feature_selection import SelectKBest, f_classif

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

def train_and_test(X, y, train, test):
    """ train and test iteration
    """
    x_train_stacked = []
    x_test_stacked = []
    coeffs = []
    y_train, y_test = y[train], y[test]
    for k in range(X.shape[2]):
        x = X[..., k]
        x_train, x_test = x[train], x[test]

        rdg = RidgeCV(alphas=np.logspace(-3, 3, 7))
        rdg.fit(x_train, y_train)
        x_train_stacked.append(rdg.predict(x_train))
        x_test_stacked.append(rdg.predict(x_test))
        coeffs.append(rdg.coef_)

    x_train_ = np.asarray(x_train_stacked).T
    x_test_ = np.asarray(x_test_stacked).T


    rfc = RandomForestClassifier()
    rfc.fit(x_train_, y_train)
    scores = rfc.score(x_test_, y_test)
    
    print 'score rfc: %.2f' % (scores)    
    B = Bunch(score=scores
              #proba=probas,
              #coeff=coeffs,
              #coeff_lgr=coeff_lgr,
              #proba_kbest=probas_kbest,
              #coeff_lgr_kbest=coeff_lgr_kbest,
              #xpred=np.concatenate((x_train_, x_test_)),
              #ypred=np.concatenate((y_train, y_test)),
              )

    return B

##############################################################################
##############################################################################
FEAT_DIR = set_features_base_dir()
CACHE_DIR = set_cache_base_dir()


dataset = fetch_adni_rs_fmri_conn('fmri_subjects_msdl_seeds.npy')
conn = np.load(dataset['fmri_data'], mmap_mode='r')

idx = set_group_indices(dataset['dx_group'])

X = conn[np.concatenate((idx['AD'], idx['EMCI'], idx['LMCI'])), ...]

from joblib import dump, load
tmp_file = os.path.join(CACHE_DIR, 'datax')
dump(X, tmp_file)
X = load(tmp_file, mmap_mode='r')

y = np.array([1] * len(idx['AD']) + [0] * len(idx['EMCI']) + \
             [0] * len(idx['LMCI']))

n_iter = 100
sss = StratifiedShuffleSplit(y, n_iter, test_size=.2,
                             random_state=np.random.seed(42))

print 'classification'

from joblib import Parallel, delayed
p = Parallel(n_jobs=-1, verbose=5)\
            (delayed(train_and_test)(X, y, train, test)\
            for train, test in sss)

np.savez_compressed(os.path.join(FEAT_DIR, 'stacking_kbest'), data=p)