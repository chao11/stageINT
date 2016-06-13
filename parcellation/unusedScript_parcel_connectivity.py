import numpy as np
import os
import  os.path as op
import nibabel as nib
import joblib as jbl
import matplotlib.pylab as plt
import nibabel as nib

subject = 'AHS22'
hemisphere = 'lh'
nb_cluster = 7
norma = 'norm2'
root_dir = '/hpc/crise/hao.c/data'


subject_dir = op.join(root_dir,subject)
tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemisphere.upper()))

output_name = subject + '_' +hemisphere.upper() + '_seed_parcellisation_cl' + str(nb_cluster) + norma +'.nii.gz'
output_path = op.join(tracto_dir,output_name)
ward_output = op.join(tracto_dir,'ward_{}_{}.jl'.format(nb_cluster,norma))
conn_matrix_path = op.join(tracto_dir,'conn_matrix_norma_{}.jl'.format(norma))

# connectivity matrix: (Mx163)
connect = jbl.load(conn_matrix_path)

# ward clustering and labels
ward_norm2 = jbl.load(ward_output)
label = ward_norm2.labels_ # M voxels

for i in np.unique(label):

    indx = np.where(label==i)
    parcel_connmatrix = connect[indx]

    plt.figure()
    plt.imshow(parcel_connmatrix,aspect='auto', interpolation= 'nearest')
    plt.colorbar()

plt.show()