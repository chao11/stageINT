# compare the size of parcellations of all the subjects

import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
import joblib
from sklearn import metrics


connect = joblib.load('/hpc/crise/hao.c/data/AHS22/tracto/LH_STS+STG_destrieux/conn_matrix_seed2parcels.jl')
matrix = connect[0]


