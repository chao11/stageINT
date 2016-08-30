#! /usr/bin/python -u
# coding=utf-8

"""
Leace p percent of subjects out, pick the subjects randomly
This script in similar to LOSO_RL_loopsubject.py
Instead of reading the connectivity matrix and functional response of each subject, it load the whole X and beta
lh_destrieux_2_All_subj_rcon_rspmT.jl stores three files:
[0]:subject
[1]:X = connectivity matrix = N_voxel*163_target,
[2]:Y = betas = N_voxel*40_beta

"""
# hao.c
# 16/07/2016


import nibabel as nib
import joblib
import os
import os.path as op
import numpy as np
from sklearn import cross_validation, linear_model
import cross_validation_LabelShuffleSplite as cv
import sys
from sklearn.metrics import r2_score
import pandas as pd
from openpyxl import load_workbook
# to fix the TclError: no display name and no $DISPLAY environment variable, set 'Agg' before import matplotlib
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pylab as plt
from modelisation import Normalization, SetModelParameter


def normaliser(x, option, subject):
#   normalize by the norm
    if option == 'norm':
        from sklearn.preprocessing import normalize
        x_norma = normalize(x, norm='l2')

#   normalize by the sum of the row, ( normalized matrix sum to 1 )
    elif option == 'sum': # normalize sum to 1:
        from sklearn.preprocessing import normalize
        x_norma = normalize(x, norm='l1')

#   normalize each row by z-score : (x-mean)/std
    elif option == 'zscore':
        from scipy import stats
        x_norma = stats.zscore(x, axis=1)
        # set the nan to 0
        x_norma[np.isnan(x_norma)] = 0

# normalize each nulber by the number of voxels of target region
    elif option == 'partarget':
        target_path = op.join(root_dir, subject, 'freesurfer_seg', target_name)
        col, target_label = lateral_model(lateral, hemisphere, target_path)
        target_mask = nib.load(target_path).get_data()

        # target_label is bilateral or ipsilateral
        # label = np.unique(target)[1:]
        target_size = []
        for i in target_label:
            size = len(target_mask[target_mask == i])
            target_size.append(size)
        # divide each number in the matrix by the size of the target region
        x_norma = np.divide(x, target_size)

    elif option == 'waytotal':
        waytotal_file = op.join(root_dir, subject, tracto_dir, 'waytotal')
        with open(waytotal_file, 'r')as f:
            file = f.read()
            text = file.split('\n')
            nb_waytotal = np.fromstring(text[0], dtype=int, sep=" ")
            # print "waytotal: ", nb_waytotal
        x_norma = x/nb_waytotal

    elif option == 'none':
       # print ('no normalization')
        x_norma = x
    return x_norma

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


def lateral_model(lateral, hemi, target_path):

    def ipsi_model(hemi, target_path):
        target_label = np.unique(nib.load(target_path).get_data())[1:]

        if hemi == 'lh':
            label = range(0, 40) + range(1001, 1036) + range(3001, 3036) + range(11101, 11176) + range(13101, 13176) + [251, 252, 253, 254, 255]
        else:
            label = range(40, 80) + range(2001, 2036) + range(4001, 4036) + range(12101, 12176) + range(14101, 14176) + [251, 252, 253, 254, 255]

        target = []
        columns = []
        # get the targets for ipsilateral:
        for index, i in enumerate(target_label):
            if i in label:
                target.append(i)
                columns.append(index)
        # print 'ipsilateral targets ', target

        return columns, target

    if lateral == 'ipsi':
        print "use the connectivity of ipsilateral " + hemi
        # take one of the target mask
        columns, target_label = ipsi_model(hemi, target_path)
    else:
        target_label = np.unique(nib.load(target_path).get_data())[1:]
        columns = range(0, len(target_label))
        print 'use the connectivity of bilateral'

    return columns, target_label


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
    # print 'MAE:', mae

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


# split the data in respect of the label(subject), return the set of X, Y and index of voxel as a dictionary
def split_and_norma_data(subject, x, y, labels, norma_option, weight, lateral, hemisphere):

    target_path = op.join(root_dir, subject[0], 'freesurfer_seg', target_name)
    col, target_label = lateral_model(lateral, hemisphere, target_path)

    set_of_x = np.empty((0, len(col)), float)
    set_of_y = np.empty(0, float)
    voxel_subject = []
#   register the index of voxel of each subject in a dictionary id = {subject, index}
    for s in labels:
        id = np.where(subject == s)[0]
        split_x = x[id, :]
        if weight == 'distance' and model != 'distance':
            distance_mat_path = op.join(root_dir, s, 'control_model_distance', '{}_distance_control_{}.jl'
                                        .format(hemisphere, parcel_altas))
            distance_mat = joblib.load(distance_mat_path)
            # 3 voxel are nan in subject AHS22
            if s == 'ACE12' and hemisphere == 'lh' and parcel_altas == 'destrieux':
                 print "remove voxels [10640,10641,10743] of subject ACE12 "
                 distance_mat = np.delete(distance_mat, [10640, 10641, 10743], 0)

            distance_mat = distance_mat[:, col]
            # print distance_mat.shape

            split_x = split_x[:, col] * distance_mat
        else:
            split_x = split_x[:, col]
        # normaliser:
        split_x = normaliser(split_x, norma_option, s)
        set_of_x = np.append(set_of_x, split_x, axis=0)
        set_of_y = np.append(set_of_y, y[id], axis=0)
        voxel_subject = voxel_subject + [s]*len(id)

    return set_of_x, set_of_y, voxel_subject


# from the predict value of one iteration, save the predict value of each subject of each iteration and get the mean score of
# output name indicate the iteration
def save_all_predict(voxel_subject_test, predict, true_y, RL_coeff, iteration):

       # save predict map for a ubject:
    def save_a_predict_img(subject, RL_coeff, subj_predict_value, subj_true_val, output_base_dir, file_name):
        coord_path = op.join(root_dir, subject, tracto_dir, 'coords_for_fdt_matrix2')
        coord = read_coord(coord_path)

        if subject == 'ACE12' and hemisphere == 'lh' and parcel_altas == 'destrieux':
            coord = np.delete(coord, [10640, 10641, 10743], 0)

        # save the predict map
        predict_nii = np.zeros((256, 256, 256))
        for i in range(len(coord)):
            c = coord[i]
            predict_nii[c[0], c[1], c[2]] = subj_predict_value[i]

        true_functional_path = op.join(fMRI_dir, subject, y_basedir)
        true_func_nii = nib.load(true_functional_path)
        img = nib.Nifti1Image(predict_nii, true_func_nii.get_affine())

        output_predict_nii_path = op.join(output_base_dir, '%s.nii.gz' % file_name)
        img.to_filename(output_predict_nii_path)

        r2 = r2_score(subj_true_val, subj_predict_value)
        output_predict_jl_path = op.join(output_base_dir, '%s.jl' % file_name)
        joblib.dump([coord, RL_coeff, subj_predict_value, r2], output_predict_jl_path, compress=3)
        return r2

    list_subject = np.unique(voxel_subject_test)
    r2_of_each_subject = {}
    # predict value of each subject:
    for subj in list_subject:
        # save the predict values into the tractodir
        predict_subject_dir = op.join(root_dir, subj, tracto_dir, 'predict', 'LPSO_%s' %(lateral))
        if not op.isdir(predict_subject_dir):
            os.mkdir(predict_subject_dir)

        outputname = '%s_Weighted_%s_%s_%s_%s_%s_predi_%s_it%.2d' % (weight, hemisphere, lateral, model, norma,
                                                                         parcel_altas, y, iteration+1)

        id_voxel = [i for i, s in enumerate(voxel_subject_test) if s == subj]

        predict_y_subject = predict[id_voxel]   # get the predict value of subject x
        true_y_subject = true_y[id_voxel] # get the true value of subject x

        # save the result
        r2_subject = save_a_predict_img(subj, RL_coeff, predict_y_subject, true_y_subject, predict_subject_dir, outputname)

        r2_of_each_subject[subj] = r2_subject

    print("iteration {}, r2_of_each_subject {}\nmean of tests:{}"
          .format(iteration, r2_of_each_subject, sum(r2_of_each_subject.itervalues())/len(r2_of_each_subject)))
    return r2_of_each_subject


def lpss_model(dict_lpss, hemisphere, data, y, norma, lateral, weight, info_file_path):

    subject = data[0]
    subjects_list = np.unique(subject)
    all_mat = data[1]

    y_col = contrast_list.index(y)
    all_Y = data[2][:, y_col]

    MAE = np.zeros(len(lpss))
    r2_iteration = np.zeros(len(lpss))
    index = np.arange(0, len(subject), 1)

    target_path = op.join(root_dir, subject[0], 'freesurfer_seg', target_name)
    col, target_label = lateral_model(lateral, hemisphere, target_path)

    it = 0
    dict_it_info = []

    for it_lpss in dict_lpss:
        info = {}
        # print ("Train:",train_index,"Test:",test_index)
        subject_test = it_lpss["test"]
        subject_train = it_lpss["train"]

    #   base de test: apply distance and ipsi
        print "leave sujects out: " + str(subject_test)
        x_test, y_test, id_test = split_and_norma_data(subject, all_mat, all_Y, subject_test, norma,
                                                       weight, lateral, hemisphere)

    #   base d'apprentissage:
        x_train, y_train, id_train = split_and_norma_data(subject, all_mat, all_Y, subject_train, norma,
                                                          weight, lateral, hemisphere)
    #   LEARN MODEL
        coeff, predi, MAE[it] = learn_model(x_train, y_train, x_test, y_test)

        r2_iteration[it] = r2_score(y_test, predi)
        print 'mean R2 score of test subjects of iteration [%d]: %f ' % (it+1, r2_iteration[it])

    #   save the predict value of each subject in ths set of test:
        r2_score_subjects = save_all_predict(id_test, predi, y_test, coeff, it)

    #   save the information in a dictionary:
        info = {'subject_train': subject_train, 'subject_test': subject_test, 'R2_score_test': r2_score_subjects,
                'R2_score_interation': r2_iteration[it]}

    #   add the info dict to list
        dict_it_info.append(info)

    #   save the list of info ;
        joblib.dump(dict_it_info, info_file_path, compress=3)

        it += 1

    print "mean R2 score:", np.mean(r2_iteration)
    return np.mean(r2_iteration), dict_it_info


# ============================ main ====================================================================================
#

hemisphere = str(sys.argv[1])
parcel_altas = str(sys.argv[2])
model = str(sys.argv[3])
y = str(sys.argv[4])
lateral = str(sys.argv[5])
weight = str(sys.argv[6])
# lpss_path = str(sys.argv[7])

lpss_path = '/hpc/crise/hao.c/model_result/tracto_volume/labelshuffle_leave_P_subject_out/leave_P_subject_out_lpss_list.jl'
lpss = joblib.load(lpss_path)
"""
hemisphere = 'lh'
parcel_altas = 'destrieux'
model = 'connmat'
y = 'rspmT_0001'
lateral='ipsi' # or bilateral
weight = 'none'

"""
# output file:
LPSS_all_inter_mean_score_file_name = 'LPSO_mean_r2_all_model.jl'

y_basedir = 'nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space/{}.nii'.format(str(y))
fMRI_dir = '/hpc/banco/voiceloc_full_database/func_voiceloc'
root_dir = '/hpc/crise/hao.c/data'
model_result_dir = '/hpc/crise/hao.c/model_result'

contrast_list = ['rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004', 'rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004']
options = ['none', 'waytotal', 'norm', 'sum', 'partarget', 'zscore']

tracto_name = '{}_STS+STG_{}_2'.format(hemisphere.upper(), parcel_altas)
tracto_dir = 'tracto_volume/%s' % tracto_name

ouput_score_dir = op.join(model_result_dir, 'tracto_volume', 'labelshuffle_leave_P_subject_out')
if not op.isdir(ouput_score_dir):
    os.mkdir(ouput_score_dir)

data_file_name = '{}_{}_2_All_subj_X_rcon_rspmT.jl'.format(hemisphere, parcel_altas)
data_path = '%s/AllData_jl/%s' %(model_result_dir, data_file_name)
data = joblib.load(data_path)
subject = data[0]
subjects_list = np.unique(subject)

#   split the subjects:
# lpss = cv.LabelShuffleSplit(subjects_list, n_iter=2, test_size=0.2)
# modeling = SetModelParameter(hemisphere, lateral, 'lpss', norma, weight, y, parcel_altas, tracto_dir)
target_name = '{}_target_mask_{}_165.nii.gz'.format(hemisphere, parcel_altas)

# compare the score for the same lpss mosel:
list_mean_score_dict = []
for norma in options:

    each_inter_info_file_name = 'r2score_{}_{}_{}_{}_{}_weighted{}_normalise_{}.jl'.\
        format(hemisphere, parcel_altas, lateral, model, y, weight, norma)
    each_inter_info_path = op.join(ouput_score_dir, each_inter_info_file_name)

    mean_r2, lpss_info = lpss_model(lpss, hemisphere, data, y, norma, lateral, weight, each_inter_info_path)

    d = {'altas': parcel_altas, 'hemisphere': hemisphere, 'model': model, 'lateral': lateral,
                     'weight': weight, 'y_file': y, 'norma': norma, 'mean_r2': mean_r2}
    print d

    list_mean_score_dict.append(d)
#   save the score
    LPSS_score_path = op.join(ouput_score_dir, LPSS_all_inter_mean_score_file_name)
    joblib.dump(list_mean_score_dict, LPSS_score_path, compress=3)


df = pd.DataFrame(list_mean_score_dict)

print 'size of dataframe: ' + str(df.shape)
print 'Max R2 score:\n{}' .format((df.max(axis=0)))



