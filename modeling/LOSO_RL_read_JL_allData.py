# This script in similar to LOSO_RL_loopsubject.py
# LOSO: leave one subject out
# Instead of reading the connectivity matrix and functional response of each subject, it load the whole X and beta
# lh_destrieux_All_subj_XYdata.jl stores three files:
# [0]:subject
# [1]:X = connectivity matrix = N_voxel*163_target,
# [2]:Y = betas = N_voxel*40_beta

# Attention: NAN values found in betas : voxel indice in ACE12 : [10640,10641,10743],
# use _All_subj_XYdata_removeNAN.jl


import joblib
import numpy as np
from sklearn import cross_validation, linear_model
import os .path as op

def normaliser(x, option):
    if option == 'l2':
        from sklearn.preprocessing import normalize
        x_norma = normalize(x, norm='l2')
#   normalize by the sum of the row, ( normalized matrix sum to 1 )
    elif option=='l1': # normalize sum to 1:
        from sklearn.preprocessing import normalize
        x_norma = normalize(x,norm='l1')

    return x_norma


def learn_model(x_train, y_train, x_test, y_test):
    # Create linear regression object
    regression = linear_model.LinearRegression()

    # Train the lodel using the training sets
    regression.fit(x_train, y_train)

    # the coefficients
    # print ('Ceofficients: \n', regression.coef_)

    # the mean absolute error MAE
    predict = regression.predict(x_test)
    mae = np.mean(abs(predict-y_test))
    print 'MAE:', mae

    return regression.coef_, predict , mae


def transform_beta2con(beta, contrast):
    beta_voix = beta[:, 0:20]
    beta_nonvoix = beta[:, 20:]

    if contrast == 1:
        con = np.mean(beta_voix, axis=1) - np.mean(beta_nonvoix, axis=1)
    elif contrast == 2:
        con = np.mean(beta_voix, axis=1) + np.mean(beta_nonvoix, axis=1)
    elif contrast == 3:
        con = np.mean(beta_voix, axis=1)
    elif contrast == 4:
        con = np.mean(beta_nonvoix, axis=1)

    # con = con*20

    return con

#
# ============================ main ====================================================================================
#
norma = 1
hemisphere = 'rh'
parcel_altas = 'destrieux'
#data_file_name = '{}_{}_All_subj_Xbeta_removeNAN.jl'.format(hemisphere, parcel_altas)
data_file_name = '{}_{}_All_subj_X_rcon_rspmT.jl'.format(hemisphere, parcel_altas)

# load data
data_path = '/hpc/crise/hao.c/model_result/AllData_jl/{}'.format(data_file_name)
data = joblib.load(data_path)
subject = data[0]
X = data[1]
beta = data[2]


# transform the beta values to contrast
con = transform_beta2con(beta, contrast=1)
# print np.where(np.isnan(con1))

subjects_list = np.unique(subject)
loso = cross_validation.LeaveOneOut(len(subjects_list))
MAE =np.zeros(len(loso))
index = np.arange(0,len(subject), 1)

# ===== split subjects for LOSO (Leave One Subject Out)=============
for train_index, test_index in loso:
    # print ("Train:",train_index,"Test:",test_index)
    print "loov: " + str(test_index)
    subject_test= [subjects_list[index] for index in test_index]
    subject_train= [subjects_list[index] for index in train_index]

#   base de test:
    print "sujet test: ", subject_test[0]
    id_test = np.where(subject == subject_test[0])[0]
    X_test = X[id_test,:]
    Y_test = con[id_test]

#   base d'apprentissage:
#    id_train = np.setdiff1d(set(index),set(id_test))
    id_train = np.where(subject != subject_test[0])[0]
    X_train = X[id_train, :]
    Y_train= con[id_train]
    # print X_train.shape, Y_train.shape

    if norma == 1:
        print "Normalize connectivity matrix. "
        X_train = normaliser(X_train)
        X_test = normaliser(X_test)
    else:
        print "Do not normalize. "

    # LEARN MODEL
    coeff, predict, MAE[test_index] = learn_model(X_train, Y_train, X_test, Y_test)

print "mean MAE:",np.mean(MAE)

