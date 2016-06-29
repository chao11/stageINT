#! /usr/bin/python -u
# coding=utf-8
"""
# hao.c
# 28/06/2016

instead of use fslmath to create the mask, we recommend to use this python scprit
this scripts creates the volumetric seed and taregt mask use desikan, destreiux and wmparc altas
wmparc2009 and wmparc.
Note: wmparc2009 parcellation file is create by mri_aparc2aseg
see more in : /hpc/crise/hao.c/python_scripts/raw_data_processing/convert_Freesurf_parcel2nii.py
"""

import nibabel
import os
import os.path as op
import numpy as np
import sys


# create seed mask for STS+STG with the label number or aparc2009(destrieux)
def mask_seed(hemisphere, altas_name, label, seed_name):
    print "create seed mask "
    altas_path = op.join(fs_parcel_dir, '%s.nii' %(altas_name))
    altas_nii = nibabel.load(altas_path)
    altas = altas_nii.get_data()

    seed_mask = np.zeros((256,256,256))

    for i in label:
        indice_seed = np.where(altas == i)
        seed_mask[indice_seed]=1

    img = nibabel.Nifti1Image(seed_mask, altas_nii.get_affine())
    output_seek_mask_path = op.join(mask_dir, '%s_%s.nii.gz'%(hemisphere, seed_name))
    img.to_filename(output_seek_mask_path)
    print 'save mask:' +output_seek_mask_path

    return output_seek_mask_path


# create target mask based on the gray matter cortical parcellation: desikan or destrieux
def mask_gm(seed_path, target_altas, output):
    print "create gray matter target: "+output
    out_put_mask_path = op.join(mask_dir, output)

    altas_path = op.join(fs_parcel_dir, '%s.nii' %target_altas)
    altas_nii = nibabel.load(altas_path)
    mask = altas_nii.get_data()
    seed_mask = nibabel.load(seed_path).get_data()
    # remove seed voxels
    indice_seed = np.where(seed_mask == 1)
    mask[indice_seed] = 0

    # remove unusable labels:
    remove_label = [0,2,4,5,7,8,14,15,16,24,28,30,31,41,43,44,46,47,60,62,63,72,77,80,85,1000, 2000]
    for i in remove_label:
        mask[mask == i] = 0

    # save target mask
    print 'aparc target:' + str(len(np.unique(mask)))
    # print np.unique(mask)

    img = nibabel.Nifti1Image(mask, altas_nii.get_affine())
    img.to_filename(out_put_mask_path)
    print("save target mask: " + out_put_mask_path)


# create target mask with white matter parcellation.
# The white matter parcellation is derived from the cortical parcellation.
# We calculate the intersection of seed region defined from wmparc2009 and the wmparc
def mask_wm( seed_label , wmparc1, wmparc2):
    print "create target mask by using %s  "%altas

    wm1_name = '%s_target_mask_%s_%s.nii.gz' %(hemi, wmparc1, seed_name)
    wm2_name = '%s_target_mask_%s_%s.nii.gz' %(hemi, wmparc2, seed_name)
    out_put_wmmask1_path = op.join(mask_dir, wm1_name)
    out_put_wmmask2_path = op.join(mask_dir, wm2_name)

    wmparc_path = op.join(fs_parcel_dir, '%s.nii' %wmparc1)
    wmparc_nii = nibabel.load(wmparc_path)
    wmparc1 = wmparc_nii.get_data()

    wmparc2_path = op.join(fs_parcel_dir, '%s.nii' %wmparc2)
    wmparc2 = nibabel.load(wmparc2_path).get_data()

    wmlabel = (np.asarray(seed_label)+2000).tolist()
    label = seed_label + wmlabel
    print(label)
    for i in label:
        indice_wm = np.where(wmparc1 == i)
        wmparc1[indice_wm] =0
        wmparc2[indice_wm] = 0

    # remove unusable labels:
    remove_label = [0,2,4,5,7,8,14,15,16,24,28,30,31,41,43,44,46,47,60,62,63,72,77,80,85,1000, 2000, 3000,4000,5001,5002]
    for i in remove_label:
        wmparc1[wmparc1 == i] = 0
        wmparc2[wmparc2 == i] = 0

    # save target mask
    print 'wmparc targets ' + str(len(np.unique(wmparc1)))
    #print np.unique(wmparc)
    img = nibabel.Nifti1Image(wmparc1, wmparc_nii.get_affine())
    img.to_filename( out_put_wmmask1_path)
    print'save target mask: %s' %out_put_wmmask1_path

    print 'wmparc targets ' + str(len(np.unique(wmparc2)))
    #print np.unique(wmparc)
    img = nibabel.Nifti1Image(wmparc2, wmparc_nii.get_affine())
    img.to_filename( out_put_wmmask2_path)
    print'save target mask: %s' %out_put_wmmask2_path




# ====================== main code ============================================================================================

hemi = 'rh'
#hemi = str(sys.argv[1])
#altas = str(sys.argv[2]) # aparc/desikan ou aparc2009/destrieux
altas = 'desikan'

seed_name = 'big_STS+STG'

target_name = '%s_target_mask_%s_%s.nii.gz' %(hemi, altas, seed_name)

root_dir = '/hpc/crise/hao.c/data'
subject_list = os.listdir(root_dir)


if seed_name=='small_STS+STG':
    seed_altas = 'destrieux'
    seed_label = [11134,11174]

elif seed_name=='big_STS+STG':
    seed_altas  ='desikan'
    seed_label = [1001, 1015, 1030, 1034]

if hemi=='rh':
    seed_label = (np.asarray(seed_label) + 1000).tolist()
print seed_label

#======================== make mask============================================================
for i in subject_list:
    mask_dir = op.join(root_dir, i, 'freesurfer_seg')
    fs_parcel_dir = op.join(root_dir, i,'parc_freesurfer')
    seed_path = op.join(mask_dir, '%s_%s.nii.gz'%(hemi, seed_name))
#   create seed mask:
    if not op.isfile(seed_path):
        mask_seed(hemi, seed_altas,  seed_label, seed_name)

#   create gray matter parcellation target mask:
    mask_gm(seed_path, altas, target_name)

#   create white matter parcellation target mask: wmparc and wmparc2009
    if seed_altas =='desikan':
        wmparc1= 'wmparc'
        wmparc2 = 'wmparc2009'

    elif seed_altas=='destrieux':
        wmparc1 = 'wmparc2009'
        wmparc2 = 'wmparc'

    mask_wm(seed_label, wmparc1, wmparc2)


print 'done! '


# ================================= check mask===========================================================================
# this is only used to check if the number of targets in the mask are the same  for all the subject
n_normal = 165
target0 = np.unique(nibabel.load('/hpc/crise/hao.c/data/AHS22/freesurfer_seg/lh_target_mask_destrieux_big_STS+STG.nii.gz').get_data())

add = []
for i in subject_list:
    mask = '/hpc/crise/hao.c/data/%s/freesurfer_seg/lh_target_mask_destrieux_big_STS+STG.nii.gz'%i
    mask_nii = nibabel.load(mask).get_data()
    affine = nibabel.load(mask).get_affine()
    n_target = len(np.unique(mask_nii))
    if n_target!= n_normal:
        add.append(i)
        diff = np.setdiff1d(np.unique(mask_nii), target0)
        print diff
        for j in diff:
            mask_nii[mask_nii == i]=0
        img = nibabel.Nifti1Image(mask_nii, affine)
        img.to_filename( mask)
        print'save target mask: %s' %mask

