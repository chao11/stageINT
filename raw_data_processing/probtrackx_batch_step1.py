#! /usr/bin/python -u
# coding=utf-8

import os
import os.path as op
import commands
import sys


#hemi = 'lh'
#altas = 'wmparc'

space = str(sys.argv[1]) #volume or surface
hemi = str(sys.argv[2])
seed_name = str(sys.argv[3])    # big_STSTG
altas = str(sys.argv[4])    # destrieux
target2_name = '%s_%s'% (altas, seed_name)  # destrieux_big_STS+STG
n_samples = str(sys.argv[5])

batchmode = 'off' # by default, just print the command but don't run it
batchmode = str(sys.argv[6]) # run = launch the command



# by default( for volumetric tractography, no additional option
surf_option=''
if space=='surface':
    exten = 'gii'
elif space == 'volume':
    exten = 'nii.gz'

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
root_dir = '/hpc/crise/hao.c/data'
fs_SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

subjectList = os.listdir(root_dir)
for subject in subjectList:

# ========================== define parameters and path =============================================
    subject_dir = op.join(root_dir,subject)

    mask_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(mask_dir,'%s_%s.%s' %(hemi, seed_name, exten))

    xfm_path = op.join(subject_dir,'freesurfer_regist','freesurfer2fa.mat')
    bedpostx_path = op.join(subject_dir,'raw_dwi.bedpostX')
    sample_path = op.join(bedpostx_path, 'merged')
    brainmask_path = op.join(bedpostx_path,'nodif_brain_mask')

    target2_path = op.join(mask_dir, '%s_target_mask_%s.nii.gz' %(hemi, target2_name))
    white_surface = op.join(subject_dir, 'surface', '%s.white.gii'%hemi)

    output_tracto_dir = op.join(subject_dir,'tracto_%s' %space, '%s_%s_%s_%s' %( hemi.upper(),  seed_name, altas, n_samples))
    mat_dot = op.join(output_tracto_dir,'fdt_matrix2.dot')


# ============================ set the option for using surface:=====================================
    if space == 'surface':
        orig_NIFTI = op.join(root_dir,subject, 'orig.nii.gz' )

#       check if orig.nii.gz exists:
        if not op.isfile(orig_NIFTI):
            print('convert orig.mgz to NIFTI')
            # the convert is in interactive mode
            fs_mri_dir = op.join(fs_SUBJECTS_DIR, subject,'mri')
            cmd = '%s/mri_convert %s/orig.mgz %s' %(fs_exec_dir, fs_mri_dir, orig_NIFTI)
            commands.getoutput(cmd)

        surf_option = '--meshspace=freesurfer --seedref=%s' %orig_NIFTI

#   check the distribution surface file for fopd option
    if not op.isfile(white_surface):
        print('create *h.white.gii')
        fs_surf_dir = op.join(fs_SUBJECTS_DIR, subject,'surf')
        cmd = '%s/mris_convert %s/%s.white %s' %(fs_exec_dir, fs_surf_dir, hemi, white_surface)
        commands.getoutput(cmd)

# =========================== tractography ===========================================================

#   if not op.isfile(op.join(output_tracto_dir, 'lookup_tractspace_fdt_matrix2.nii.gz')):
    if not op.isfile(mat_dot) and not op.isfile(op.join(output_tracto_dir,'fdt_matrix2.zip')):

        #commands.getoutput('rm -rf %s' %output_tracto_dir)

        cmd =   'fsl5.0-probtrackx2 -x %s --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P %s \
        --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=%s \
        --forcedir --opd -s %s -m %s --dir=%s --omatrix2 --target2=%s --fopd=%s  %s' \
                %(seed_path, n_samples, xfm_path, sample_path, brainmask_path, output_tracto_dir, target2_path, white_surface, surf_option)

        """
        cmd =  'fsl5.0-probtrackx2 -x %s --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 5000 \
        --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=%s \
        --forcedir --opd -s %s -m %s --dir=%s --omatrix2 --target2=%s'%(seed_path,xfm_path, bedpostx_path, mask_path, output_tracto_dir, target_path)
        """

        print cmd

        if batchmode=='run':
            batch_cmd = "frioul_batch '%s' " %cmd
            commands.getoutput(batch_cmd)

    else:
        print subject +' done!'

