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


def learn_model(x_train, y_train, x_test, y_test):

    # Create linear regression object
    regression = linear_model.LinearRegression()

    # Train the lodel using the training sets
    regression.fit(x_train, y_train)

    # the coefficients
    #print ('Ceofficients: \n', regression.coef_)

    # the mean absolute error MAE
    predict = regression.predict(x_test)
    mae = np.mean(abs(predict-y_test))
    print 'MAE:', mae

    return regression.coef_, predict , mae


# read the coordinates of seed =========================================================================================
def read_coord(file_path):

    with open(file_path,'r') as f:
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
def extract_functional(coord, img_fmri):
    y = []
    for i in coord:
        y.append(img_fmri[i[0], i[1], i[2]])
    return y


# get the connectivity matrix and the functional response ==============================================================
def get_data(x_path, y_path, seed_coord):
#
    connmat = joblib.load(x_path)
    if 'distance' in x_path:
        x = connmat
    else:
        x = connmat[0]

    y_nii = nib.load(y_path)
    y_img = y_nii.get_data()
    #extract y values:
    y = np.asarray(extract_functional(seed_coord, y_img))

    return x,y


def normaliser(x):
  # Nomaliser:
    from sklearn.preprocessing import normalize
    x_norma = normalize(x,norm='l2')
    return x_norma


def remove_nan(x, y):
    # NAN value may be found in some voxels, remove them from the dataset
    # remove nan and update data
    nan_ind = np.unique(np.where(np.isnan(y))[0])
    if len(nan_ind)>0:
        print "find NAN in dataset, remove and update"

        x = np.delete(x, nan_ind, 0)
        y = np.delete(y, nan_ind, 0)
    return x, y, nan_ind


def loso_model(hemisphere, parcel_altas, model, y_file, norma):

    """
    subjects_list = os.listdir(root_dir)
    fMRI_dir = '/hpc/banco/voiceloc_full_database/func_voiceloc'
    y_basedir = 'nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space/{}.nii'.format(str(y_file))
    tracto_dir = 'tracto/{}_STS+STG_{}/'.format(hemisphere.upper() ,parcel_altas)
    """
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
    """
    # set the model: dixtance comtrol model or connectivity matrix modeling:
    if model=='distance':
    #
        filename = op.join('control_model_distance','{0}_distance_control_{1}.jl'.format(hemisphere, parcel_altas))
    #
    else:
    #
        filename = op.join(tracto_dir, 'conn_matrix_seed2parcels.jl')

    print "modeling %s altas: %s, \nY :%s, \nX:%s, norma:%s \n" %(hemisphere, parcel_altas, y_file, filename, str(norma))
    """
    # set the number of target:
    #
    if parcel_altas == 'destrieux':
    #
        nb_target = 163  # 165 for new target mask
    #
    elif parcel_altas=='aparcaseg':
    #
        nb_target = 87
    #
    elif parcel_altas == 'wmparc':
    #
        nb_target = 155


    # remove the subject which doesn't have rspmT
    for i in subjects_list[:]:
    #
        y_test_path = op.join(fMRI_dir, i, y_basedir)
    #
        if not op.isfile(y_test_path):
    #
            print( y_test_path + "  not exist")
    #
            subjects_list.remove(i)
    print("length of the list: " + str(len(subjects_list)))


    loov = cross_validation.LeaveOneOut(len(subjects_list))
    print "loov:", len(loov)

    MAE =np.zeros(len(loov))
    #
    # ===== splite subjects for LOOV=============
    for train_index, test_index in loov:
    #
    #   print ("Train:",train_index,"Test:",test_index)
        print "loov: " + str(test_index)
        subject_test= [subjects_list[index] for index in test_index]
        subject_train= [subjects_list[index] for index in train_index]

    #   base de test:
        print "sujet test: " ,subject_test[0]
        x_test_path = op.join(root_dir, subject_test[0], filename)
        y_test_path = op.join(fMRI_dir, subject_test[0],y_basedir )

        test_coord_path = op.join(root_dir,subject_test[0],tracto_dir,'coords_for_fdt_matrix2')
        test_seed_coord = read_coord(test_coord_path)
        print("load X_test: " + x_test_path + "\n load Y_test: " + y_test_path )
        X_test, Y_test = get_data(x_test_path, y_test_path, test_seed_coord)

    #   base d'apprentissage:
        # print("train:", subject_train)
        X_train = np.empty((0,nb_target), float)
        Y_train = np.empty(0, float)
#
        for subject in subject_train:
            x_train_path = op.join(root_dir, subject, filename)
            y_train_path = op.join(fMRI_dir, subject,y_basedir )
#
            train_coord_path = op.join(root_dir, subject,tracto_dir,'coords_for_fdt_matrix2')
            train_seed_coord = read_coord(train_coord_path)
#
            X, Y = get_data(x_train_path,y_train_path, train_seed_coord)
            X_train = np.append(X_train, X, axis=0)
            Y_train = np.append(Y_train, Y, axis=0)

        print X_train.shape, Y_train.shape


    #   check for nan values:
        X_test,Y_test, test_nan_id = remove_nan(X_test,Y_test)
        X_train, Y_train, nan_train  = remove_nan(X_train, Y_train)

        if len(test_nan_id)>0:
            print "remove the seed voxel:", test_nan_id
            test_seed_coord = np.delete(test_seed_coord, test_nan_id, 0)

    #   normalization:
        if norma == 1:
            print "normaliser la matrice"
            X_train = normaliser(X_train)
            X_test = normaliser(X_test)
        else:
            print "non normaliser"

    #   LEARN MODEL
        coeff, predi, MAE[test_index] = learn_model(X_train, Y_train, X_test, Y_test)

#
        # save the predict values
        predict_subject = op.join(root_dir, subject_test[0], 'predict')
        if not op.isdir(predict_subject):
            os.mkdir(predict_subject)

        predict_output = op.join(predict_subject, '{}_{}_predi_{}.jl'.format(hemisphere, model, y_file))

        joblib.dump([test_seed_coord,predi], predict_output, compress=3)

        # save the predict map
        predict_nii = np.zeros((256,256,256))
        for i in range(len(test_seed_coord)):
            c = test_seed_coord[i]
            predict_nii[c[0],c[1],c[2]] = predi[i]

        img = nib.Nifti1Image(predict_nii, nib.load(y_test_path).get_affine())
        img_output_path = op.join(predict_subject, '{}_{}_predi_{}.nii'.format(hemisphere, model, y_file))
        img.to_filename(img_output_path)

    print "mean MAE:", np.mean(MAE)

    plt.plot(MAE)
    plt.title('MAE of %s_%s_%s, mean: %s' %(hemisphere, model, y_file, str(np.mean(MAE))))
    plt.savefig('/hpc/crise/hao.c/model_result/{}_{}_{}_{}_norma{}.png'.format(hemisphere,parcel_altas, model, y_file, str(norma)))

    return np.mean(MAE)


# ==================== main============================================================

hemisphere = ['lh','rh']
parcel_altas = ['aparcaseg', 'wmparc']
model = 'distance'
y_file = ['rspmT_0001','rspmT_0002','rspmT_0003','rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']

"""
hemisphere = str(sys.argv[1])
parcel_altas = str(sys.argv[2])
model = str(sys.argv[3])
y_file = str(sys.argv[4])
"""
norma = 1


root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
fMRI_dir = '/hpc/banco/voiceloc_full_database/func_voiceloc'

dlist = []
for p in parcel_altas:
    for h in hemisphere:
        for y in y_file:

            y_basedir = 'nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space/{}.nii'.format(str(y))
            tracto_dir = 'tracto/{}_STS+STG_{}/'.format(h.upper(),p)

            if model=='distance':
            #
                filename = op.join('control_model_distance','{0}_distance_control_{1}.jl'.format(h, p))
            #
            else:
            #
                filename = op.join(tracto_dir, 'conn_matrix_seed2parcels.jl')

            print "modeling %s altas: %s, \nY :%s, \nX:%s, norma:%s \n" %(h, p, y, filename, str(norma))

            mean_MAE = loso_model(h, p, model, y, norma)
            d = {'altas': p, 'hemisphere': h, 'model':model, 'y_file' :y, 'mean_MAE': mean_MAE}

            dlist.append(d)
print dlist