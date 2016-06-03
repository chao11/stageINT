#! /usr/bin/python -u
# coding=utf-8

# hao.c
# 09/05/2016
# To predict function from connectivity, we use a leave-one-subject-out cross validation routine.
# We use a linear regressing to model the relationship between the fMRI contrast response and the connectivity matrix.
# We predict the left-out subject by applying the coefficients to the connectivity matrix of the subject andcalculate the absolute erros


import nibabel as nib
import joblib
import os
import os.path as op
import numpy as np
from sklearn import cross_validation, linear_model
import sys
import matplotlib.pylab as plt


def learn_model(Xtrain, Ytrain, Xtest, Ytest):

    # Create linear regression object
    regression = linear_model.LinearRegression()

    # Train the lodel using the training sets
    regression.fit(Xtrain, Ytrain)

    # the coefficients
    #print ('Ceofficients: \n', regression.coef_)

    # the mean absolute error MAE
    predict = regression.predict(Xtest)
    mae = np.mean(abs(predict-Ytest))
    print 'MAE:', mae

    return regression.coef_, predict , mae


# read the coordinates of seed =========================================================================================
def read_coord(coord_file_path):

    with open(coord_file_path,'r') as f:
        file = f.read()
        text = file.split('\n')
        # print("total number of voxels:", len(text)-1)

        seed_coord = np.zeros((len(text)-1, 3), dtype= int)

        for i in range(len(text)-1):
        #
            data = np.fromstring(text[i], dtype=int, sep=" ")
        #
            seed_coord[i, 0] = data[0]
        #
            seed_coord[i, 1] = data[1]
        #
            seed_coord[i, 2] = data[2]

    return seed_coord


# check if all the coordinate in coord file correspond ot the seed mask ================================================
def check_coords(seed_coord, mask):
    check = []
    for i in seed_coord:
        check.append(mask[i[0], i[1], i[2]])

    print("unique :",np.unique(check))

    if np.unique(check)==1:
        return True

    # if there are some incorrect coordinate
    else:
        print "find incorrect coordinates !!!"
        incorrect =  seed_coord[check.index(0)]
        return incorrect


# extract the functionnal response for each voxel correspondent ========================================================
def extract_functional(coord, img_fMRI):
    y = []
    for i in coord:
        y.append(img_fMRI[i[0], i[1], i[2]])
    return y


# get the connectivity matrix and the functional response ==============================================================
def get_data(connmat_path, rsmpT_path, coord_file_path):
    connmat = joblib.load(connmat_path)
    if 'distance' in connmat_path:
        X = connmat
    else:
        X = connmat[0]


    rspmT= nib.load(rsmpT_path)
    rsmp = rspmT.get_data()
    #extract y values:
    coord = read_coord(coord_file_path)
    Y= np.asarray(extract_functional(coord, rsmp))

    return X,Y

def normaliser(X):
  # Nomaliser:
    from sklearn.preprocessing import normalize
    X_norma = normalize(X,norm='l2')
    return X_norma



# ==================== main=================================================================================================

hemisphere = 'lh'
parcel_altas = 'destrieux'
contrast = 1
norma = 1
filename = 'conn_matrix_seed2parcels.jl'
#filename = 'distance_control'
"""
hemisphere = str(sys.argv[1])
parcel_altas = str(sys.argv[2])
contrast = str(sys.argv[3])

# filename = str(sys.argv[4])
filename = 'conn_matrix_seed2parcels.jl'

#norma = str(sys.argv[5])
norma = 1
"""
print "modeling %s altas: %s, contrast:%s,file:%s, norma:%s " %(hemisphere, parcel_altas, contrast, filename, str(norma))

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
fMRI_dir = '/hpc/banco/voiceloc_full_database/func_voiceloc'
rsmpT_basedir = 'nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space/rcon_000{}.nii'.format(str(contrast))
tracto_dir = 'tracto/{}_STS+STG_{}/'.format(hemisphere.upper() ,parcel_altas)

# check if the coordinate of the seed is correct
"""
for subject in subjects_list:
    # reac coordinate file:
    coord_file_path = op.join(root_dir,subject,tracto_dir,'coords_for_fdt_matrix2')
    coord = read_coord(coord_file_path)

    # check seed mask
    seed_path = op.join(root_dir,subject,'freesurfer_seg','{}_STS+STG.nii.gz'.format(hemisphere.lower()))
    seed = nib.load(seed_path)
    mask = seed.get_data()
    check_coords(coord,mask)
"""


# remove the subject which doesn't have rspmT
for i in subjects_list[:]:
    rsmpT_test_path = op.join(fMRI_dir, i, rsmpT_basedir)
    if not op.isfile(rsmpT_test_path):
        print(i + " rspmT not exist")
        subjects_list.remove(i)
print("length of the list: " + str(len(subjects_list)))


loov = cross_validation.LeaveOneOut(len(subjects_list))
print "loov:", len(loov)

MAE =np.zeros(len(loov))
#===== splite subjects for LOOV=============
for train_index, test_index in loov:
   # print ("Train:",train_index,"Test:",test_index)
    print "loov: " + str(test_index)
    subject_test= [subjects_list[index] for index in test_index]
    subject_train= [subjects_list[index] for index in train_index]

#   base de test:
    print "sujet test: " ,subject_test[0]
    connmat_test_path = op.join(root_dir, subject_test[0],tracto_dir, filename)
    rsmpT_test_path = op.join(fMRI_dir, subject_test[0],rsmpT_basedir )
    coord_file_path = op.join(root_dir,subject_test[0],tracto_dir,'coords_for_fdt_matrix2')
    X_test, Y_test = get_data(connmat_test_path,rsmpT_test_path,coord_file_path)

#   base d'apprentissage:
    # print("train:", subject_train)
    X_train = np.empty((0,163), float)
    Y_train = np.empty(0, float)

    for subject in subject_train:
        connmat_train_path = op.join(root_dir, subject,tracto_dir, filename)
        rsmpT_train_path = op.join(fMRI_dir, subject,rsmpT_basedir )
        coord_file_path = op.join(root_dir, subject,tracto_dir,'coords_for_fdt_matrix2')

        X, Y= get_data(connmat_train_path,rsmpT_train_path, coord_file_path)
        X_train = np.append(X_train, X, axis=0)
        Y_train = np.append(Y_train, Y, axis=0)

    # print X_train.shape, Y_train.shape

    if norma == 1:
        print "normaliser la matrice"
        X_train = normaliser(X_train)
        X_test = normaliser(X_test)
    else:
        print "non normaliser"
    # LEARN MODEL
    coeff, predi, MAE[test_index] = learn_model(X_train, Y_train, X_test, Y_test)


print "mean MAE:",np.mean(MAE)

plt.plot(MAE)
plt.title('MAE of %s_%s_%s, mean: %s' %(hemisphere,filename, str(contrast), str(np.mean(MAE))))
plt.savefig('/hpc/crise/hao.c/model_result/{}_{}_{}_norma{}.png'.format(hemisphere, filename, str(contrast), str(norma)))
