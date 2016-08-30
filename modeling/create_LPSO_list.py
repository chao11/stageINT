#
"""
create the dictionary of test and train subjects for  leave P subject out model

"""
import joblib
import numpy as np
from sklearn import cross_validation, linear_model
import cross_validation_LabelShuffleSplite as cv
import commands as cmd
import os.path as op

hemisphere = 'lh'
parcel_altas = 'destrieux'
model = 'connmat'
y = 'rspmT_0001'
lateral='ipsi' # or bilateral
weight = 'none'

n_iter = 50
test_size = 0.2

model_result_dir = '/hpc/crise/hao.c/model_result'
ouput_score_dir = op.join(model_result_dir, 'tracto_volume', 'labelshuffle_leave_P_subject_out')

data_file_name = '{}_{}_2_All_subj_X_rcon_rspmT.jl'.format(hemisphere, parcel_altas)
data_path = '%s/AllData_jl/%s' %(model_result_dir, data_file_name)
data = joblib.load(data_path)
subject = data[0]
subjects_list = np.unique(subject)

#   split the subjects:
lpss = cv.LabelShuffleSplit(subjects_list, n_iter=n_iter, test_size=test_size)

lpss_dict = []
for train_index, test_index in lpss:

    # print ("Train:",train_index,"Test:",test_index)
    subject_test = [subjects_list[index] for index in test_index]
    subject_train = [subjects_list[index] for index in train_index]
    #print "leave sujects out: " + str(subject_test)
    lpss_dict.append({"test": subject_test, "train": subject_train})


for i in lpss_dict:
    subject_test = i["test"]
    print subject_test

print lpss_dict

save_lpss_path = op.join(ouput_score_dir, 'leave_%s_subjects_out_lpss_list.jl' %(str(len(lpss))))

joblib.dump(lpss_dict, save_lpss_path, compress=3)
print save_lpss_path


cmd = "python /hpc/crise/hao.c/python_scripts/modeling/Leave_P_out_RL.py {hemi} {altas} {model} {y} {lateral} {weight} {lpss}"\
    .format(hemi=hemisphere, altas=parcel_altas, model=model, y=y, lateral=lateral, weight=weight, lpss=save_lpss_path)
print cmd
