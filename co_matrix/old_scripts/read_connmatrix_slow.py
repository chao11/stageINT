#! /usr/bin/python -u
# load the fdt_matrix.dot file (2.5~3G) and compute the connectivity matrix. this method takes more than 5 hours or even more and need large memory for calculate

import numpy as np
from scipy import sparse
import nibabel as nb
import os
import os.path as op
import joblib

import sys

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
hemisphere = str(sys.argv[1])

for subject in subjects_list[1:2]:

    # define directories
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    target_path = op.join(fs_seg_dir,'target_mask.nii.gz')
    output_tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemisphere.upper()))
    fdt_fullmatrix_path = op.join(output_tracto_dir,'fdt_matrix2.dot')
    output_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')

    if op.isfile(output_path):
        print 'subject %s conn_matrix exits' %subject

    else:
        #print subject
        print('Processing subject {}...'.format(subject))

        # load full matrix! takes a long time an requires a lot of memory (need frioul01!)
        a = np.loadtxt(fdt_fullmatrix_path)

        conn_matrix = sparse.coo_matrix((a[:,2], (a[:,0]-1,a[:,1]-1)))
        print('{}: full matrix floaded!!'.format(subject))

        # load target file
        target_nii = nb.load(target_path)
        target_data = target_nii.get_data()
        print target_data.shape

        # load seed, target coordinates
        seed_coords_path = op.join(output_tracto_dir,'coords_for_fdt_matrix2')
        seed_coords = np.loadtxt(seed_coords_path).astype(np.int)
        n_seed_voxels = seed_coords.shape[0]

        target_coords_path = op.join(output_tracto_dir,'tract_space_coords_for_fdt_matrix2')
        target_coords = np.loadtxt(target_coords_path).astype(np.int)
        n_target_voxels = target_coords.shape[0]
        print "number of targets",n_target_voxels

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

        # save matrix in disk
        output_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')
        joblib.dump([conn_matrix_parcelated,labels],output_path,compress=3)
        print('{}: saved reduced connectivity matrix!!'.format(subject))
        print "the shape of the connectivity matrix is ",conn_matrix_parcelated.shape

        # joblib.load( conn_matrix_seed2parcels) type:list
