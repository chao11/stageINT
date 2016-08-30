#! /usr/bin/python -u
# coding=utf-8
"""
compute the distance between each seed voxel and the centre of targets
in order to get the same size as connectivity matrix and the same order of voxels, use coords_for_fdt_matrix2

"""
import os
import os.path as op
import numpy as np
import nibabel as nib
import joblib
import sys


# read the coordinates of seed =========================================================================================
# the coordinate are the seme for
def read_coord(coord_file_path):

    with open(coord_file_path,'r') as f:
        file = f.read()
        text = file.split('\n')
        # print("total number of voxels:", len(text)-1)

        coord = np.zeros((len(text)-1, 3), dtype= int)

        for i in range(len(text)-1):
        #
            data = np.fromstring(text[i], dtype=int, sep=" ")
        #
            coord[i, 0] = data[0]
        #
            coord[i, 1] = data[1]
        #
            coord[i, 2] = data[2]

    return coord


# calculate the center of mass of the targets
def target_center(target_path):

    target = nib.load(target_path)
    target_img = target.get_data()

    # determine the unique regions
    labels = np.unique(target_img)[1:]
    print 'number of targets:', len(labels)
    center_coords = []
    for label in labels:
        # calculate the center of mass
        center_coords.append(np.mean(np.asarray(np.nonzero(target_img == label)),axis=1))

    return center_coords


# get the information ( target mask, seed mask etc) from probtrackx.log
def get_info_from_log(tracto_dir, flag):
    log_file = op.join(tracto_dir, 'probtrackx.log')
    with open(log_file, 'r') as f:
        probtackLog = f.read()
        inputs = probtackLog.split(' ')
        for i in inputs:
            if flag in i:
                info = i.replace(flag, '')
    return info


# =================================== main =============================================================================
tracto_name = 'tracto_volume/LH_small_STS+STG_destrieux_WM_5000'

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list:

    tracto_dir = op.join(root_dir, subject, tracto_name)
    coord_file_path = op.join(tracto_dir,  'coords_for_fdt_matrix2')
    target_path = get_info_from_log(tracto_dir=tracto_dir, flag='--target2=')
    target_file = op.basename(target_path)
    output_filename = target_file.replace('target_mask','distance_control')
    output_filename = output_filename.replace('nii.gz', 'jl')
    output_path = op.join(root_dir, subject,'control_model_distance', output_filename)

    if not op.isfile(output_path):
        # load target mask and seed coordinates
        print "calculating %s's  distance control model: %s" %(subject, target_path)
        center_coords = target_center(target_path)

        print("get seed coordinates: ", coord_file_path)
        seed_coord = read_coord(coord_file_path)
        print "the number of seed vertex/voxel:", seed_coord.shape

        # calculate the distance between each voxel andthe center of the target
        dist_mat = np.zeros((seed_coord.shape[0], len(center_coords)))
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                dist_mat[i, j] = np.sqrt(np.sum((seed_coord[i]-center_coords[j])**2))
        print dist_mat.shape

        # get output name and seva in joblib
        joblib.dump(dist_mat, output_path, compress=3)
        print "save %s\n" % output_path
    else:
        print "%s done" % subject