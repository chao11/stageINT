#! /usr/bin/python -u
# coding=utf-8

# hao.c
# instead of use fslmath to create the mask, we recommend to use this python scprit

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


# create target mask based on the gray matter cortical parcellation: aparc(aparcaseg) or aparc2009(destrieux)
def mask_GM(seed_path, target_altas, target_name):
    print "create gray matter target: "+target_name
    mask_path = op.join(mask_dir, target_name)
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
    img.to_filename(mask_path)
    print("save target mask: " + mask_path)

# create target mask with white matter parcellation.
# The white matter parcellation is derived from the cortical parcellation.
# We calculate the intersection of seed region defined from wmparc2009 and the wmparc
def mask_WM(subject, hemisphere ):
    print "create target mask by using wmparc for "+ subject+ hemisphere
    wmparc_path = '/hpc/crise/hao.c/data/%s/parc_freesurfer/wmparc.nii' %subject
    wmparc_nii = nibabel.load(wmparc_path)
    wmparc = wmparc_nii.get_data()
    print 'total number of altas label: '+ str(len(np.unique(wmparc)))
    # mask_aparc(altas, seed_mask)

    wmparc2009 = nibabel.load('/hpc/crise/hao.c/data/%s/parc_freesurfer/wmparc2009.nii'%subject).get_data()
    if hemisphere=='lh':
        indice_wm = np.where((wmparc2009 ==13134) | (wmparc2009 ==13174) | (wmparc2009 ==11134) | (wmparc2009 ==11174)  )
    elif hemisphere=='rh':
        indice_wm = np.where((wmparc2009 ==14134) | (wmparc2009 ==14174) | (wmparc2009 ==12134) | (wmparc2009 ==12174)  )
    else:
        print'hemisphere unknown'

    wmparc[indice_wm]=0

    # remove unusable labels:
    remove_label = [0,2,4,5,7,8,14,15,16,24,28,30,31,41,43,44,46,47,60,62,63,72,77,80,85,1000, 2000, 3000,4000,5001,5002]
    for i in remove_label:
        wmparc[wmparc==i]=0

    # save target mask
    print 'wmparc target ' + str(len(np.unique(wmparc)))
    #print np.unique(wmparc)
    img = nibabel.Nifti1Image(wmparc, wmparc_nii.get_affine())
    img.to_filename( '/hpc/crise/hao.c/data/%s/freesurfer_seg/%s_target_mask_wmparc.nii.gz' %(subject, hemisphere))


# ======================main============================================================================================

#hemi = 'rh'
hemi = str(sys.argv[1])
altas = str(sys.argv[2]) # aparc ou aparc2009/destrieux
#altas = 'aparc+aseg'

seed_name = 'big_STS+STG'

target_name = '%s_target_mask_%s_%s.nii.gz' %(hemi, altas, seed_name)

root_dir = '/hpc/crise/hao.c/data'
subject_list = os.listdir(root_dir)


if seed_name=='small_STS+STG':
    seed_altas = 'destrieux'
    seed_label = [11134,11174]
elif seed_name=='big_STS+STG':
    seed_altas  ='aparc+aseg'
    seed_label = [1001, 1015, 1030, 1034]

if hemi=='rh':
    seed_label = np.asarray(seed_label)+1000
print seed_label

#======================== make mask============================================================
for i in subject_list:
    mask_dir = op.join(root_dir, i, 'freesurfer_seg')
    fs_parcel_dir = op.join(root_dir, i,'parc_freesurfer')
    seed_path = op.join(mask_dir, '%s_%s.nii.gz'%(hemi, seed_name))

#   create seed mask:
    if not op.isfile(seed_path):
        mask_seed(hemi, seed_altas,  seed_label, seed_name)

#   create target mask:
    mask_GM(seed_path, altas, target_name)

#   mask_aparc(i,hemi, 'aparc')
print 'done! '


