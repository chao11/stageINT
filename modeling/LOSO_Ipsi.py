"""
Leave one subject out model
use the connectivity of ipsilateral hemisphere

"""


import nibabel as nib
import joblib
import os
import os.path as op
import numpy as np
from sklearn import cross_validation, linear_model
import sys
import matplotlib.pylab as plt
import commands
from sklearn.metrics import r2_score
import pandas as pd
from openpyxl import  load_workbook

hemisphere = 'lh'
parcel_altas = 'destrieux'

data_file_name = '{}_{}_All_subj_X_rcon_rspmT.jl'.format(hemisphere, parcel_altas)

# load data
data_path = '/hpc/crise/hao.c/model_result/AllData_jl/{}'.format(data_file_name)
data = joblib.load(data_path)
subject = data[0]
X = data[1]
rcon2 = data[2][:, 1]

subjects_list = np.unique(subject)
loso = cross_validation.LeaveOneOut(len(subjects_list))
MAE =np.zeros(len(loso))
index = np.arange(0,len(subject), 1)



label = joblib.load('/hpc/crise/hao.c/data/ACE12/tracto/LH_STS+STG_destrieux_2/conn_matrix_seed2parcels.jl')[1]


target_label = {}
target_label= {'destrieux_small_STS+STG': label, 'desikan_small_STS+STG':label}