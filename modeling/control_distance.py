#! /usr/bin/python -u
# coding=utf-8

# compute the distance between each voxel and the centre of targets

import os
import os.path as op
import numpy as np
import nibabel as nib
import joblib
import time
import sys

# read the coordinates of seed =========================================================================================
# the coordinate are the seme for
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

# calculate the center of mass of the targets
def target_center(target_path):
    target = nib.load(target_path)
    target_img = target.get_data()

    #determine the unique regions
    labels = np.setdiff1d(np.unique(target_img.ravel()), [0])
    center_coords = []
    for label in labels:
        # calculate the center of mass
        center_coords.append(np.mean(np.asarray(np.nonzero(target_img==label)),axis=1))

    return center_coords

# =================================== main =============================================================================

#hemisphere = 'lh'
#parcel_alta = 'destrieux'

hemisphere = str(sys.argv[1])
parcel_alta = str(sys.argv[2])
target_base_name = str(sys.argv[3])
tracto_name = str(sys.argv[4]) # tracto/.... or tracto_surface_


root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

target_name = '{}_target_mask_{}.nii.gz'.format(hemisphere, target_base_name)  # _destrieux_big_STS+STG
output_filename = '{}_distance_control_{}.jl'.format(hemisphere, target_base_name)

for subject in subjects_list:

    target_path = op.join(root_dir, subject, 'freesurfer_seg', target_name)
    output_path = op.join(root_dir, subject,'control_model_distance', output_filename)

    if not op.isfile(output_path):
        print "calculatingg %s %s %s distance control model" %(subject, hemisphere, parcel_alta)
        # load target mask and seed coordinates

        center_coords = target_center(target_path)

        coord_file_path = op.join(root_dir, subject, tracto_name,  'coords_for_fdt_matrix2')
        seed_coord = read_coord(coord_file_path)

        # calculate the distance between each voxel andthe center of the target
        dist_mat = np.zeros((seed_coord.shape[0],len(center_coords)))
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                dist_mat[i,j] = np.sqrt(np.sum((seed_coord[i]-center_coords[j])**2))
        print dist_mat.shape

        joblib.dump(dist_mat, output_path, compress=3)
        print "model saved\n"