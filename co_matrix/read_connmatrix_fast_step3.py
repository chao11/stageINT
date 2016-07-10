#! /usr/bin/python -u
# coding=utf-8

"""
Read the fdt_matrix2.dot  and trancform to the M*N connectivity matrix
This script is uesd for one subject, instead of using numpy.loadtxt, read the connectivity matrix line by line

parameters:
    - hemisphere = lh/rh ,
    - tarcto_parcel: freesurfer parcellation for probtrackX (desterieux, aparcaseg, wmparc)

use: target mask, fdt_matrix2.dot(zip), coords_for_fdt_matrix2 , tract_space_coords_for_fdt_matrix2

output:  conn_matrix_seed2parcels.jl
"""

import os
import os.path as op
import numpy as np
from scipy import sparse
import nibabel as nib
import joblib
import time
import sys
import zipfile

hemisphere = 'lh'
# subject = 'AHS22'
# seed = '{}_big_STS+STG.gii'.format(hemisphere.lower())
target = 'destrieux_big_STS+STG'

"""
hemisphere = str(sys.argv[1])
tracto_parcel = str(sys.argv[2])
target =  str(sys.argv[3])
"""
root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

tracto_name = 'tracto_surface/LH_small_STS+STG_destrieux_5000'
# tracto_dir = '/hpc/crise/hao.c/data/AHS22/tracto_surface/LH_big_STS+STG_destrieux_500'

for subject in subjects_list:

    subject_dir = op.join(root_dir, subject)
    fs_seg_dir = op.join(subject_dir, 'freesurfer_seg')
    # seed_path = op.join(fs_seg_dir, seed)

    target_name = '{}_target_mask_{}.nii.gz'.format(hemisphere, target)
    target_path = op.join(fs_seg_dir, target_name)

    tracto_dir = op.join(subject_dir, tracto_name)
    fdt_fullmatrix_path = op.join(tracto_dir, 'fdt_matrix2.zip')
    output_path = op.join(tracto_dir, 'conn_matrix_seed2parcels.jl')

    if not op.isfile(fdt_fullmatrix_path):
        print "%s not exists" % fdt_fullmatrix_path

    elif op.isfile(output_path):
        print " %s connectivity matrix exists" % subject

    else:

        # load and compute cnnectivity matrix
        print "load connectivity matrix %s:" % fdt_fullmatrix_path
        t2 = time.time()

        # connectivity_matrix_file = open(fdt_fullmatrix_path)
        z = zipfile.ZipFile(fdt_fullmatrix_path)
        filename = z.namelist()[0]
        f = z.open(filename, "r")

        # connectivity_matrix = z.read(filename)

        connectivity_matrix = f.read()
        text = connectivity_matrix.split('\n')

        z.close()

        A = np.zeros((len(text), 3))
    #
        print 'compute connectivity matrix...'
    #
    #   Convert the format : X Y #number_of_tracts (matrice A)  -->  matrix #voxel_seed times #voxel_targets (matrice connect)
        for i in range(len(text)-1):
        #
            data = np.fromstring(text[i], dtype=int, sep=" ")
        #   in python, index starts from 0
            A[i, 0] = data[0]-1
        #
            A[i, 1] = data[1]-1
        #
            A[i, 2] = data[2]
        #

        # defining sparse matrix from A : connect is the connectivity matrix #seed X #targets

    #   matrice de connectivite de la forme seed voxels x target voxels,
        # number of rows and number of columns correspond to the last line of the matrix (last line in fdt-matrix.dot)
        # coo_matrix (data, (i,j))), data[:]the entiers fo the matrix, i,j the row/column indices of the entries
        conn_matrix = sparse.coo_matrix((A[:, 2], (A[:, 0], A[:, 1])), dtype=np.float32)
    #
        print(' full matrix loaded!!')

        # load target file

        target_nii = nib.load(target_path)
        target_data = target_nii.get_data()

        # load seed, target coordinates
        seed_coords_path = op.join(tracto_dir, 'coords_for_fdt_matrix2')
        seed_coords = np.loadtxt(seed_coords_path).astype(np.int)
        n_seed_voxels = seed_coords.shape[0]

        target_coords_path = op.join(tracto_dir, 'tract_space_coords_for_fdt_matrix2')
        target_coords = np.loadtxt(target_coords_path).astype(np.int)
        n_target_voxels = target_coords.shape[0]
        print n_target_voxels

        target_labels = np.zeros(n_target_voxels, dtype=np.int)
        for i in range(n_target_voxels):
            target_labels[i] = target_data[target_coords[i, 0], target_coords[i, 1], target_coords[i, 2]]

        # convert to Compressed Sparse Column type, to perform fast artithmetics on columns
        c = conn_matrix.tocsc()
        labels = np.unique(target_labels)
        n_target_parcels = labels.size
        print(n_target_parcels)
        conn_matrix_parcelated = np.zeros([n_seed_voxels, n_target_parcels])

        for label_ind, current_label in enumerate(labels):
            target_inds = np.where(target_labels == current_label)[0]
            conn_matrix_parcelated[:, label_ind] = c[:, target_inds].sum(1).squeeze()

        print "shape of connectivity matrix:", conn_matrix_parcelated.shape

        # save matrix in disk
        joblib.dump(conn_matrix_parcelated, output_path, compress=3)
        print('{}: saved reduced connectivity matrix!!'.format(subject))
        print "time: %s" % str(time.time()-t2)
