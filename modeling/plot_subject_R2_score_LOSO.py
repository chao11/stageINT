"""
plot R2 score of a set of parameter in order of subject
"""

import joblib
import os
import os.path as op
import numpy as np
import pandas as pd
import xlrd
import nibabel as nib
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pylab as plt


def getFileName(path):
    f_list = os.listdir(path)
    jl_list = []
    for i in f_list:
        if op.splitext(i)[1] == '.jl':
            print i
            jl_list.append(i)

    print len(jl_list)
    return jl_list


model_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_aparcaseg'
jl_score_list = getFileName(model_score_path)

for jl in jl_score_list:
    print jl
    score = joblib.load(op.join(model_score_path, jl))
    print score.shape
    if 'AHS22' in score.index:
        score = score.drop('AHS22')

    mean_score = score.mean(axis=0)
    print mean_score
    score.plot(figsize=(20, 10))
    plt.xlabel('subject')
    plt.ylabel('R2 score')
    plt.legend(loc='lower left')
    plt.title(jl)
    plt.text(50, round(np.amin(score.min(axis=0))), 'mean of R2 score: \n {}'.format(mean_score))

    plt.show()
    plt.savefig(op.join(model_score_path, '%s.png' %jl))






