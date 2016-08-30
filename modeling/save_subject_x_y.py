# hao.c 02/06/2016
# loop over all the subject and save the connectivity matrix and 40 resampled beta values in disk

import nibabel as nib
import joblib
import os
import os.path as op
import numpy as np



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


# extract the functionnal response for each voxel correspondent ========================================================
def extract_y(coord, y_img):
    y = []
    for i in coord:
        y.append(y_img[i[0], i[1], i[2]])
    return np.array(y)


# get the y values: beta or contrast:
def get_y(coordinates, base_dir, option):

    if option =='beta':
        y = np.zeros((len(coordinates), 40))
#
        for i in range(1, 41):
            beta_img= nib.load(op.join(base_dir, 'rbeta_%.4d.nii'%i)).get_data()

            #extract beta values:
            y[:,i-1]= np.asarray(extract_y(coordinates, beta_img))
#
    elif option=='contrast':
        y = np.zeros((len(coordinates), 8))

#       load rcon_i images
        for i in range(1,5):
            contrast_img = nib.load(op.join(base_dir, 'rcon_%.4d.nii'%i)).get_data()
            y[:,i-1]= np.asarray(extract_y(coordinates, contrast_img))
#       load rspmT_i images
        for j in range(1, 5):
            contrast_img = nib.load(op.join(base_dir, 'rspmT_%.4d.nii'%j)).get_data()
            y[:, j+3] = np.asarray(extract_y(coordinates, contrast_img))

    print 'Y shape',y.shape
    return y


# ===================================== main ===========================================================================
hemisphere = 'rh'
parcel_altas = 'aparcaseg'
y_data_option = 'contrast'

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
fMRI_dir = '/hpc/banco/voiceloc_full_database/func_voiceloc'
tracto_dir = 'tracto_volume/{}_STS+STG_{}_2/'.format(hemisphere.upper() ,parcel_altas)
y_base = 'nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space'

output = '/hpc/crise/hao.c/model_result/AllData_jl/%s_%s_All_subj_X_rcon_rspmT.jl'%(hemisphere, parcel_altas)

# remove the subject which doesn't have rspmT
for i in subjects_list[:]:
    beta_path = op.join(fMRI_dir, i, y_base)
    if not op.isdir(beta_path):
        print(i + " rspmT not exist")
        subjects_list.remove(i)
print("length of the list: " + str(len(subjects_list)))

if y_data_option == 'beta':
    Y = np.empty((0, 40), float)
else:
    Y = np.empty((0, 8), float)

index_subject = np.empty(0, str)
mat = joblib.load(op.join(root_dir, subjects_list[0], tracto_dir, 'conn_matrix_seed2parcels.jl'))
print op.join(root_dir, subjects_list[0], tracto_dir, 'conn_matrix_seed2parcels.jl')
X = np.empty((0, mat.shape), float)


for subject in subjects_list:
    connmat_path = op.join(root_dir, subject,tracto_dir, 'conn_matrix_seed2parcels.jl')
    y_base_dir = op.join(fMRI_dir, subject, y_base)
    coord_file_path = op.join(root_dir, subject, tracto_dir, 'coords_for_fdt_matrix2')
    coord = read_coord(coord_file_path)

    connmat = joblib.load(connmat_path)
    x = connmat[0]
    y = get_y(coord, y_base_dir, y_data_option)
    sub = np.repeat(subject,len(coord),axis=0)

    index_subject = np.append(index_subject, sub, axis=0)
    X = np.append(X, x, axis=0)
    Y = np.append(Y, y, axis=0)

# NAN may be exist in beta because of the resampling
# check if there are nan valeus in beta and remove them
nan_ind = np.unique(np.where(np.isnan(Y))[0])
if len(nan_ind)>0:
    print "find NAN in dataset, remove and update"
    index_subject = np.delete(index_subject, nan_ind, 0)
    X = np.delete(X, nan_ind, 0)
    Y = np.delete(Y, nan_ind, 0)
else:
    print"no NAN values"

joblib.dump([index_subject, X], output, compress=3)
joblib.dump([index_subject, Y], '/hpc/crise/hao.c/model_result/AllData_jl/%s_All_subj_rcon_rspmT.jl'%(hemisphere), compress=3)

print"data saved in ", output