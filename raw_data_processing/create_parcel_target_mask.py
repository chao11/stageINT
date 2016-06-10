# hao.c
# instead of use fslmath to create the mask, we recommend to use this python scprit



import nibabel
import os
import os.path as op
import numpy as np
import sys

# create seed mask for STS+STG with the label number or aparc2009(destrieux)
def mask_seed(subject, hemisphere):
    print "create seed mask " + subject
    altas_path = '/hpc/crise/hao.c/data/%s/parc_freesurfer/destrieux.nii' %subject
    altas_nii = nibabel.load(altas_path)
    altas = altas_nii.get_data()

    seed_mask = np.zeros((256,256,256))
    if hemisphere == 'lh':
        seed_label = [11134,11174]
    else:
        seed_label = [12134, 12174]

    for i in seed_label:
        indice_seed = np.where(altas == i)
        seed_mask[indice_seed]=1

    img = nibabel.Nifti1Image(seed_mask, altas_nii.get_affine())
    img.to_filename('/hpc/crise/hao.c/data/%s/freesurfer_seg/%s_STS+STG.nii.gz' %(subject, hemisphere))


# create target mask based on the cortical parcellation: aparc or aparc2009(destrieux)
def mask_aparc(subject, hemisphere, altas):
    print "create target mask by using " + altas +' for ' + subject + hemisphere

    if altas in ['aparc', 'aparcaseg']:
        altas_path = '/hpc/crise/hao.c/data/%s/parc_freesurfer/aparc+aseg.nii' %subject
    elif altas in ['aparc2009', 'destrieux']:
        altas_path = '/hpc/crise/hao.c/data/%s/parc_freesurfer/destrieux.nii' %subject

    seed_path = '/hpc/crise/hao.c/data/%s/freesurfer_seg/%s_STS+STG.nii.gz' %(subject, hemisphere)
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
    img.to_filename('/hpc/crise/hao.c/data/%s/freesurfer_seg/%s_target_mask_%s.nii.gz' %(subject, hemisphere, altas))


# create target mask with white matter parcellation.
# The white matter parcellation is derived from the cortical parcellation.
# We calculate the intersection of seed region defined from wmparc2009 and the wmparc
def mask_wmparc(subject, hemisphere ):
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

hemi = 'rh'

#hemi = str(sys.argv[1])
#altas = str(sys.argv[2]) # aparc ou aparc2009/destrieux

subject_list = os.listdir('/hpc/crise/hao.c/data')
for i in subject_list:
    # create seed mask:
    # mask_seed(i,hemi)

    # create target mask:
    mask_wmparc(i,hemi)
    # mask_aparc(i,hemi, 'aparc')
print 'done! '


