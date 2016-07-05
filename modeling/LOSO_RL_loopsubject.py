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
import commands
from sklearn.metrics import r2_score
import pandas as pd
from openpyxl import load_workbook


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
def get_data( x_path, y_path, seed_coord):
    connmat = joblib.load(x_path)
    if 'distance' in x_path:
        x = connmat
    else:
        x = connmat[0]

    y_nii = nib.load(y_path)
    y_img = y_nii.get_data()

    # extract y values:
    y = np.asarray(extract_functional(seed_coord, y_img))
    return x, y


def normaliser(x, option):
  # normalize by the norm
    if option=='norm':
       # print 'normalize by the norm or the row'
        from sklearn.preprocessing import normalize
        x_norma = normalize(x,norm='l2')

#   normalize by the sum of the row, ( normalized matrix sum to 1 )
    elif option=='sum': # normalize sum to 1:
        #print('normalize by the sum of the row')
        from sklearn.preprocessing import normalize
        x_norma = normalize(x,norm='l1')

#   normalize each row by z-score : (x-mean)/std
    elif option=='zscore':
        from scipy import stats
        x_norma = stats.zscore(x, axis=1)
        # set the nan to 0
        x_norma[np.isnan(x_norma)]=0

    elif option=='none':
       # print ('no normalization')
        x_norma = x

    return x_norma


def norma_par_target(mat, target_file):
    print "target path:", target_file
    target = nib.load(target_file).get_data()
    label = np.unique(target)[1:]
    target_size = []
    for i in label:
        size = len(target[target == i])
        target_size.append(size)

    # divide each number in the matrix by the size of the target region
    norma_mat = np.divide(mat, target_size)
    return norma_mat


def remove_nan(x, y):
    # NAN value may be found in some voxels, remove them from the dataset
    # remove nan and update data
    nan_ind = np.unique(np.where(np.isnan(y))[0])
    if len(nan_ind)>0:
        print "find NAN in dataset, remove and update"

        x = np.delete(x, nan_ind, 0)
        y = np.delete(y, nan_ind, 0)
    return x, y, nan_ind


 #   get the number of waytatal:
def get_waytotal(file_path):
    with open(file_path, 'r')as f:
        file = f.read()
        text = file.split('\n')
        waytotal = np.fromstring(text[0], dtype=int, sep=" ")
        print "waytotal: ", waytotal
    return waytotal


def lateral_model(lateral, hemi, target_path):

    def ipsi_model(hemi, target_path):

        print 'use only the connection probabilities of ' + hemi
        target_label = np.unique(nib.load(target_path).get_data())[1:]

        if hemi == 'lh':
            label = range(0, 40) + range(1001, 1036) + range(3001, 3036) + range(11101, 11176) + range(13101, 13176) + [251, 252, 253, 254, 255]
        else:
            label = range(40, 80) + range(2001, 2036) + range(4001, 4036) + range(12101, 12176) + range(14101, 14176) + [251, 252, 253, 254, 255]

        target = []
        columns = []
        for index, i in enumerate(target_label):
            if i in label:
                target.append(i)
                columns.append(index)
        print 'ipsilateral targets ', target

        return columns, target

    if lateral == 'ipsi':
        print "use the connectivity of ipsilateral " + hemisphere
        # take one of the target mask
        col, target_label = ipsi_model(hemisphere, target_path)
    else:
        target_label = np.unique(nib.load(target_path).get_data())[1:]
        col = range(0, len(target_label))
        print 'use the connectivity of bilateral'

    return col, target_label


def loso_model(list, hemisphere, parcel_altas, model, y_file, norma, lateral):

    target_path = op.join(root_dir, list[0], 'freesurfer_seg', target_name)
    # for ipsilateral modeling:

    loov = cross_validation.LeaveOneOut(len(list))
    print "loov:", len(loov)

    MAE =np.zeros(len(loov))
    r2 = np.zeros(len(loov))
    #
    # ===== splite subjects for LOOV=============
    for train_index, test_index in loov:
    #
        # print ("Train:",train_index,"Test:",test_index)
        print "loov: " + str(test_index)
        subject_test = [list[index] for index in test_index]
        subject_train = [list[index] for index in train_index]

#      base de test:==========================================================================================
        print "sujet test: ", subject_test[0]
        x_test_path = op.join(root_dir, subject_test[0], filename)
        y_test_path = op.join(fMRI_dir, subject_test[0], y_basedir)

        test_coord_path = op.join(root_dir,subject_test[0], tracto_dir, 'coords_for_fdt_matrix2')
        test_seed_coord = read_coord(test_coord_path)
        print("load X_test: " + x_test_path + "\n load Y_test: " + y_test_path )
        X_test, Y_test = get_data(x_test_path, y_test_path, test_seed_coord)

      #   multiple by the distance weighted matrix
        if multip_distance == 1:
            distance_mat_path = op.join(root_dir, subject_test[0], 'control_model_distance','{0}_distance_control_{1}.jl'.format(hemisphere, parcel_altas))
            distance_mat = joblib.load(distance_mat_path)
            X_test = X_test[:, col] * distance_mat[:, col]
        else:
            X_test = X_test[:, col]

    #   normalize
        if norma == 'waytotal':
            waytotal_file = op.join(root_dir, subject_test[0], tracto_dir, 'waytotal')
            nb_waytotal = get_waytotal(waytotal_file)
            X_test = X_test/nb_waytotal

        elif norma == 'partarget':
            X_test = norma_par_target(X_test, target_file=op.join(root_dir, subject_test[0], 'freesurfer_seg', target_name))

        else:
            X_test = normaliser(X_test, norma)

#      base d'apprentissage:==================================================================================
        # print("train:", subject_train)
        nb_target = X_test.shape[1]
        X_train = np.empty((0, nb_target), float)
        Y_train = np.empty(0, float)
#
        for subject in subject_train:
            x_train_path = op.join(root_dir, subject, filename)
            y_train_path = op.join(fMRI_dir, subject,y_basedir )
#
            train_coord_path = op.join(root_dir, subject, tracto_dir,'coords_for_fdt_matrix2')
            train_seed_coord = read_coord(train_coord_path)
#
            X, Y = get_data(x_train_path,y_train_path, train_seed_coord)

#           normalize
            if norma == 'waytotal':
                waytotal_file = op.join(root_dir, subject, tracto_dir, 'waytotal')
                X = X/get_waytotal(waytotal_file)

            elif norma == 'partarget':
                X = norma_par_target(X, target_file=op.join(root_dir, subject, 'freesurfer_seg', target_name))
            else:
                X = normaliser(X, norma)

            X_train = np.append(X_train, X, axis=0)
            Y_train = np.append(Y_train, Y, axis=0)

        print X_train.shape, Y_train.shape


#       check for nan values:
        X_test, Y_test, test_nan_id = remove_nan(X_test,Y_test)
        X_train, Y_train, nan_train  = remove_nan(X_train, Y_train)

        if len(test_nan_id)>0:
            print "remove the seed voxel:", test_nan_id
            test_seed_coord = np.delete(test_seed_coord, test_nan_id, 0)


#       LEARN MODEL
        coeff, predi, MAE[test_index] = learn_model(X_train, Y_train, X_test, Y_test)
        r2[test_index] = r2_score(Y_test, predi)
        print 'r2: ', r2[test_index]

        # save the predict values
        predict_subject = op.join(root_dir, subject_test[0], 'predict')
        if not op.isdir(predict_subject):
            os.mkdir(predict_subject)

        predict_output = op.join(predict_subject, '{}_{}_{}_{}_{}_predi_{}.jl'.format(hemisphere, lateral, model, norma, parcel_altas,  y_file))
        joblib.dump([test_seed_coord, predi], predict_output, compress=3)

#       save the predict map
        predict_nii = np.zeros((256,256,256))
        for i in range(len(test_seed_coord)):
            c = test_seed_coord[i]
            predict_nii[c[0], c[1], c[2]] = predi[i]

        img = nib.Nifti1Image(predict_nii, nib.load(y_test_path).get_affine())
        predict_output_path = op.join(predict_subject, '{}_{}_{}_{}_{}_predi_{}.nii.gz'.format(hemisphere, lateral, model, norma, parcel_altas, y_file))
        img.to_filename(predict_output_path)

    print "mean MAE:", np.mean(MAE)

    print "mean R2 score:", np.mean(r2)

    return MAE, r2, list


# ==================== main============================================================
"""
hemisphere = 'rh'
parcel_altas = 'destrieux'
model = 'connmat'
#y_file = ['rspmT_0001','rspmT_0002','rspmT_0003','rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']
y = 'rspmT_0002'
lateral='ipsi' # or bilateral

"""
hemisphere = str(sys.argv[1])
parcel_altas = str(sys.argv[2])
model = str(sys.argv[3])
y= str(sys.argv[4])
lateral = str(sys.argv[5])

multip_distance = 1
# tracto_dir = str(sys.argv[6])
#target_name = str(sys.argv[7])

tracto_name = '{}_STS+STG_{}_2'.format(hemisphere.upper(),parcel_altas)
target_name = '{}_target_mask_{}_165.nii.gz'.format(hemisphere, parcel_altas)

options = ['none', 'waytotal', 'norm', 'sum', 'partarget', 'zscore']

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'

root_dir = '/hpc/crise/hao.c/data'
fMRI_dir = '/hpc/banco/voiceloc_full_database/func_voiceloc'

output_predict_score_excel_file = '/hpc/crise/hao.c/model_result/tracto_volume/%s_LOSO_R2score_compare_normalization_%s_%s.xlsx' %(tracto_name, lateral, model)

tracto_dir = 'tracto/%s' % tracto_name


subjects_list = os.listdir(root_dir)

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

y_basedir = 'nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space/{}.nii'.format(str(y))

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

# ============================= compare result of each normalization methode ===========================================
dict_R2 = {}
dlist = []

for norma in options:

    if model == 'distance':
        filename = op.join('control_model_distance','{0}_distance_control_{1}.jl'.format(hemisphere, parcel_altas))

    else:
        filename = op.join(tracto_dir, 'conn_matrix_seed2parcels.jl')

    print "modeling %s altas: %s, \nY :%s, \nX:%s, norma:%s \n" %(hemisphere, parcel_altas, y, filename, str(norma))

#   modeling:
    mae, R2, sub_list = loso_model(subjects_list, hemisphere, parcel_altas, model, y, norma, lateral)
    dict_R2 [norma] = pd.Series(R2, index=sub_list)

#   mean score of this set of LOSO
    d = {'norma':norma,  'y_file' :y, 'mean_MAE': np.mean(mae), 'R2':np.mean(R2)}
    dlist.append(d)

print dlist

result_dataframe = pd.DataFrame(dict_R2)

# ================================================ save score =========================================================
if op.isfile(output_predict_score_excel_file):
    book = load_workbook(output_predict_score_excel_file)
    writer = pd.ExcelWriter(output_predict_score_excel_file, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
else:
    writer = pd.ExcelWriter(output_predict_score_excel_file, engine='openpyxl')
#   save the predict file to a new worksheet
result_dataframe.to_excel(writer, y)
writer.save()

# ================================================ plot score =========================================================
result_dataframe.plot()
plt.title('r2 %s_%s_%s_%s' %(hemisphere, parcel_altas, model, y))
plt.ylabel('r2')
plt.xlabel('subject')
plt.savefig('/hpc/crise/hao.c/model_result/r2score_{}_{}_{}_{}_{}.png'.format(hemisphere, parcel_altas, lateral,model, y))

