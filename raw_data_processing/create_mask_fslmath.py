"""
this script use fslmaths to extract the seed mask and target mask, this method is not recommended

"""
import os
import os.path as op
import commands
import numpy as np
import nibabel
root_dir = '/hpc/crise/hao.c/data'
subjectList = os.listdir(root_dir)

# subjectList = ['EHN03','DMS30']
for subject in subjectList[0:1]:

    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')


    # ===========================definirr le mask seed===================================================================
    cmd_lh_seed = """fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 11134 -thr 11134 %s/lh_STG.nii.gz
fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 11174 -thr 11174 %s/lh_STS.nii.gz
cd %s
fsl5.0-fslmaths lh_STG.nii.gz -add lh_STS.nii.gz lh_STS+STG.nii.gz
fsl5.0-fslmaths lh_STS+STG.nii.gz -bin lh_STS+STG.nii.gz""" %(subject_dir,fs_seg_dir,subject_dir,fs_seg_dir,fs_seg_dir)
   # print cmd_lh_seed
    #commands.getoutput(cmd_lh_seed)

    cmd_rh_seed = """fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 12134 -thr 12134 %s/rh_STG.nii.gz
fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 12174 -thr 12174 %s/rh_STS.nii.gz
cd %s
fsl5.0-fslmaths rh_STG.nii.gz -add rh_STS.nii.gz rh_STS+STG.nii.gz
fsl5.0-fslmaths rh_STS+STG.nii.gz -bin rh_STS+STG.nii.gz""" %(subject_dir,fs_seg_dir,subject_dir,fs_seg_dir,fs_seg_dir)
    #print cmd_rh_seed
    #commands.getoutput(cmd_rh_seed)





    # ===========================definir le mask target===================================================================
    # the target mask is defined on 3 altas:  aparc+aseg.2009s;  aparc+aseg.nii; wmparc.nii (label 3000,4000,5001,5002)
    #

    sub_th_aparc2009 = [[0,8], [14,16], [24,24], [28,47], [60,85], [1000,1000], [2000,2000]]
    sub_th_aparcaseg = [[0,8], [14,16], [24,24], [28,47], [60,85], [1000,1000], [1001,1001],[1008,1008] ,[1015,1015], [1030,1030],[2000,2000], [2001,2001],[2008,2008] ,[2015,2015], [2030,2030]]
    sub_th_wmparc = [ [0,8], [14,16], [24,24], [28,47], [60,85], [1000,1000], [2000,2000], [3000,3000], [4000,4000] ,[5001,5002]]

    seed_aparc = [[2001,2001],[2008,2008] ,[2015,2015], [2030,2030]]

    parcel_altas = 'aparc+aseg.nii'

    print "cd %s" %subject_dir
    commands.getoutput("cd %s"%subject_dir)

    # extract the no-needed regions
    for th in sub_th_aparcaseg:
        cmd = "fsl5.0-fslmaths parc_freesurfer/{0} -thr {1} -uthr {2} freesurfer_seg/sub{3}.nii.gz".format(parcel_altas, str(th[0]), str(th[1]), str(sub_th_aparcaseg.index(th)+1))
        print cmd
        commands.getoutput(cmd)

    # combine those regions
    add = "fsl5.0-fslmaths freesurfer_seg/sub1.nii.gz"
    for i in range(len(sub_th_aparcaseg)-1):
        add += ' -add freesurfer_seg/sub{}.nii.gz'.format(i+2)
    add = add + ' freesurfer_seg/sub_target.nii.gz'
    print add

    #andseed = add + ' -add freesurfer_seg/lh_STS.nii.gz -add freesurfer_seg/lh_STG.nii.gz -add freesurfer_seg/rh_STG.nii.gz -add freesurfer_seg/rh_STS.nii.gz freesurfer_seg/sub_target.nii.gz'
    #print andseed

    # create target mask
    cmdtarget ='fsl5.0-fslmaths parc_freesurfer/%s -sub freesurfer_seg/sub_target.nii.gz freesurfer_seg/target_mask_2_%s'%(parcel_altas, parcel_altas)
    print cmdtarget
    commands.getoutput(cmdtarget)

    print "%s done" %subject

    print len(np.unique(nibabel.load('{0}/freesurfer_seg/target_mask_2_{1}'.format(subject_dir, parcel_altas)).get_data()))