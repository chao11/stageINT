# nNormalisation of the connectivity matrix
# ======================================================================================================================
# options for normalisation: 1. 'none': without normalisation (default)
#                            2. 'norm1': normalisation by l2
#                            3. 'standard': standardize the feature by removing the mean and scaling to unit variance
#                            4. 'range': MinMaxScaler, scale the features between a givin minimun and maximum, often between (0,1)
# ======================================================================================================================

import joblib
import numpy as np
import os
import os.path as op
import matplotlib.pylab as plt


def nomalisation(connect,norma, axis=1):
    # AXIS =1 by default
    if axis ==0:    # normalize the feature of the matrix
        connect = np.transpose(connect)

    if norma=='norm1':
        from sklearn.preprocessing import normalize
        connect_norm = normalize(connect,norm='l1')
        #connect = connect_norm


    elif norma=='norm2':
        from sklearn.preprocessing import normalize
        connect_norm = normalize(connect,norm='l2')
        #connect = connect_norm


    elif norma == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(connect)
        connect_norm = scaler.transform(connect)
        #connect = connect_scaled


    elif norma == 'MinMax':
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        connect_norm = min_max_scaler.fit_transform(connect)

    if axis == 0:
        connect_norm = connect_norm.T

    return connect_norm



root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
#hemisphere = str(sys.argv[1])
hemisphere = 'lh'
options = ['norm1','norm2','standard','MinMax']
axis = 0

for option in options:

  #  plt.figure(indx1)
    for subject in subjects_list:
        print "processing {},{}".format(subject, option)
        subject_dir = op.join(root_dir,subject)
        tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemisphere.upper()))
        conn_matrix_path = op.join(tracto_dir,'conn_matrix_seed2parcels.jl')

        conn_matrix_jl= joblib.load(conn_matrix_path)
        conn_matrix = conn_matrix_jl[0]

        matrix_norma = nomalisation(conn_matrix,norma= option ,axis=axis)

     # save matrix:
        output_path = op.join(tracto_dir,'conn_matrix_norma_{}_{}.jl'.format(option,axis))
        joblib.dump(matrix_norma,output_path,compress=3)
        print('{}: saved {} normalised connectivity matrix!!'.format(subject, option))

