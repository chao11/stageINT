#! /usr/bin/python -u
# coding=utf-8

import os
import os.path as op
import commands
import sys


#hemi = 'lh'
#altas = 'wmparc'

hemi = str(sys.argv[1])
altas = str(sys.argv[2])
nb_target = str(sys.argv[3])
batchmode = 'off'

batchmode = str(sys.argv[4])

root_dir = '/hpc/crise/hao.c/data'
SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

subjectList = os.listdir(root_dir)
for subject in subjectList:


    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(fs_seg_dir,'%s_STS+STG.nii.gz' %(hemi))

    xfm_path = op.join(subject_dir,'freesurfer_regist','freesurfer2fa.mat')
    bedpostx_path = op.join(subject_dir,'raw_dwi.bedpostX','merged')
    mask_path = op.join(subject_dir,'raw_dwi.bedpostX','nodif_brain_mask')

    #target_path = op.join(fs_seg_dir,'target_mask.nii')
    target_mask_name = '%s_target_mask_%s_%s.nii.gz' %(hemi, altas, nb_target)
    target_path = op.join(fs_seg_dir, target_mask_name)

    #    output_tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemi.upper()))
    output_tracto_dir = op.join(subject_dir,'tracto','%s_STS+STG_%s_2'%(hemi.upper(),altas))

    mat_dot = op.join(output_tracto_dir,'fdt_matrix2.dot')


    if not op.isfile(mat_dot) and not op.isfile(op.join(output_tracto_dir,'fdt_matrix2.zip')):

        #commands.getoutput('rm -rf %s' %output_tracto_dir)

        cmd =   "frioul_batch 'fsl5.0-probtrackx2 -x %s --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 5000 \
        --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=%s \
        --forcedir --opd -s %s -m %s --dir=%s --omatrix2 --target2=%s'" %(seed_path,xfm_path, bedpostx_path, mask_path, output_tracto_dir, target_path)
        """

        cmd =  'fsl5.0-probtrackx2 -x %s --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 5000 \
        --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=%s \
        --forcedir --opd -s %s -m %s --dir=%s --omatrix2 --target2=%s'%(seed_path,xfm_path, bedpostx_path, mask_path, output_tracto_dir, target_path)
        """

        print cmd

        if batchmode=='on':
            commands.getoutput(cmd)



