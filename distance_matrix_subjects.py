# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:51:25 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from fetch_data import fetch_adni_petmr, fetch_adni_masks,\
                       fetch_adni_rs_fmri_conn, set_group_indices
from nilearn.masking import apply_mask
from sklearn.manifold import MDS
import matplotlib.pylab as pl


FIG_PATH = '/home/mr243268/bib/my_papers/petmr_analysis/figures/pcc/'

mask = fetch_adni_masks()

#dataset = fetch_adni_petmr()
#x = apply_mask(dataset['pet'], mask['mask_petmr'])

dataset = fetch_adni_rs_fmri_conn('fmri_subjects.npy')

idx = set_group_indices(dataset['dx_group'])



conn = np.load(dataset['fmri_data'], mmap_mode='r')


for k in range(conn.shape[2]):

    print k
    x = conn[:, :, k]
    
    groups = [['AD', 'Normal'],
              ['AD', 'EMCI', 'LMCI'],
              ['Normal', 'EMCI', 'LMCI']]
    
    for gr in groups:
        indices = np.array([], dtype=np.int)
        for g in gr:
            indices = np.append(indices, idx[g])
        x1 = x[indices, :]
        mds = MDS(n_components=2, max_iter=5000, n_jobs=1, verbose=0,
                  eps=1e-12, random_state=42)
        s = mds.fit_transform(x1)
        print gr, 'stress: ', mds.stress_/(x1.shape[0]*x1.shape[1])
        pl.figure()
        pl.scatter(s[:len(idx[gr[0]]), 0], s[:len(idx[gr[0]]), 1],
                   s=80, c='#ee0000', marker='v')
        pl.scatter(s[len(idx[gr[0]]):, 0], s[len(idx[gr[0]]):, 1],
                   s=50, c='#00de00', marker='o')
        pl.title('/'.join(gr))
        pl.grid('on')
        pl.legend(gr)
        pl.tight_layout()
        pl.savefig(FIG_PATH + 'mds_' + str(k) + '_' + '_'.join(gr) + '.png')
        pl.savefig(FIG_PATH + 'mds_' + str(k) + '_' + '_'.join(gr) + '.pdf')
