# instead of using numpy.loadtxt, read the connectivity matrix line by line

import os
import os.path as op
import numpy as np
from scipy import sparse
import nibabel as nib
import joblib
import time

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
#hemisphere = str(sys.argv[1])

tarcto_parcel = 'aparcaseg'
hemisphere='lh'


for subject in subjects_list[1:2]:

    # define directories
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    target_path = op.join(fs_seg_dir,'target_mask.nii.gz')
    seed_path = op.join(fs_seg_dir,'{}_STS+STG.nii.gz'.format(hemisphere.lower()))
    tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_{}'.format(hemisphere.upper(),tarcto_parcel))
    fdt_fullmatrix_path = op.join(tracto_dir,'fdt_matrix2.dot')
    output_path = op.join(tracto_dir,'conn_matrix_seed2parcels.jl')

    # load and compute cnnectivity matrix
    print "load connectivity matrix fdt_matrix2.dot:"
    t2 = time.time()
    connectivity_matrix_file = open(fdt_fullmatrix_path)
    connectivity_matrix = connectivity_matrix_file.read()
    text = connectivity_matrix.split('\n')

    A = np.zeros((len(text),3))
#
    print 'compute connectivity matrix...'
#
# Convert the format : X Y #number_of_tracts (matrice A)  -->  matrix #voxel_seed times #voxel_targets (matrice connect)
    for i in range(len(text)-1):
#
        data = np.fromstring(text[i], dtype=int, sep=" ")
#
        A[i,0]=data[0]-1
#
        A[i,1]=data[1]-1
#
        A[i,2]=data[2]
#

    #defining sparse matrix from A : connect is the connectivity matrix #seed X #targets

#   matrice de connectivite de la forme seed voxels x target voxels
    conn_matrix = sparse.coo_matrix((A[:,2],(A[:,0],A[:,1])),dtype=np.float32)
#
    print('{}: full matrix floaded!!'.format(subject))

    # load target file
    target_nii = nib.load(target_path)
    target_data = target_nii.get_data()
    print target_data.shape

    # load seed, target coordinates
    seed_coords_path = op.join(tracto_dir,'coords_for_fdt_matrix2')
    seed_coords = np.loadtxt(seed_coords_path).astype(np.int)
    n_seed_voxels = seed_coords.shape[0]

    target_coords_path = op.join(tracto_dir,'tract_space_coords_for_fdt_matrix2')
    target_coords = np.loadtxt(target_coords_path).astype(np.int)
    n_target_voxels = target_coords.shape[0]

    target_labels = np.zeros(n_target_voxels,dtype=np.int)
    for i in range(n_target_voxels):
        target_labels[i] = target_data[target_coords[i,0],target_coords[i,1],target_coords[i,2]]


    # convert to Compressed Sparse Column type, to perform fast artithmetics on columns
    c = conn_matrix.tocsc()
    labels = np.unique(target_labels)
    n_target_parcels = labels.size
    conn_matrix_parcelated = np.zeros([n_seed_voxels,n_target_parcels])
    for label_ind, current_label in enumerate(labels):
        target_inds = np.where(target_labels==current_label)[0]
        conn_matrix_parcelated[:,label_ind] = c[:,target_inds].sum(1).squeeze()

    print "shape of connectivity matrix:", conn_matrix_parcelated.shape

    # save matrix in disk
    joblib.dump([conn_matrix_parcelated,labels],output_path,compress=3)
    print('{}: saved reduced connectivity matrix!!'.format(subject))

