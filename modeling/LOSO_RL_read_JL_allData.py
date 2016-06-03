# This script in similar to LOSO_RL_loopsubject.py

# Instead of reading the connectivity matrix and functional response of each subject, it load the whole X and beta
# lh_destrieux_All_subj_XYdata.jl stores three files:
# [0]:subject
# [1]:X = connectivity matrix = N_voxel*163_target,
# [2]:Y = betas = N_voxel*40_beta

# Attention: in _All_subj_XYdata.jl, NAN values found in betas : voxel indice in ACE12 : [10640,10641,10743],
# use _All_subj_XYdata_removeNAN.jl


import joblib
import numpy as np
from sklearn import cross_validation, linear_model


def normaliser(x):
    from sklearn.preprocessing import normalize
    x_norma = normalize(x, norm='l2')
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

# load data
workpath = '/hpc/crise/hao.c/model_result/{}_{}_All_subj_XYdata_removeNAN.jl'.format(hemisphere, parcel_altas)
data = joblib.load(workpath)
subject = data[0]
X = data[1]
beta = data[2]


# NAN value may be found in some voxels, remove them from the dataset
# remove nan:
nan_ind = np.unique(np.where(np.isnan(beta))[0])
if len(nan_ind)>0:
    print "find NAN in dataset, remove and update"
    subject = np.delete(subject, nan_ind, 0)
    X = np.delete(X, nan_ind, 0)
    beta = np.delete(beta, nan_ind, 0)

    output = '/hpc/crise/hao.c/model_result/%s_%s_All_subj_XYdata_removeNAN.jl'%(hemisphere, parcel_altas)
    joblib.dump([subject, X, beta],output ,compress=3)



# transform the beta values to contrast
con = transform_beta2con(beta, contrast=1)
# print np.where(np.isnan(con1))

subjects_list = np.unique(subject)
loov = cross_validation.LeaveOneOut(len(subjects_list))
MAE =np.zeros(len(loov))
index = np.arange(0,len(subject), 1)

# ===== split subjects for LOOV=============
for train_index, test_index in loov:
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