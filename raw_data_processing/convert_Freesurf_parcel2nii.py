# CONVERT FREESURFER MRI PARCELLATION TO NIFTI FOR FSL

import numpy as np
import nibabel
import commands
import os

root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)
os.chdir('/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db')


for subjectCode in subjectlist:

    os.chdir('/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/%s/' %subjectCode)
    cmd = 'mri_convert aparc+aseg.mgz /hpc/crise/hao.c/data/%s/parc_freesurfer/aparc+aseg.nii' % subjectCode
    #print cmd
    #commands.getoutput(cmd)


    cmd = 'mri_convert wmparc.mgz /hpc/crise/hao.c/data/%s/parc_freesurfer/wmparc.nii' % subjectCode
    #print cmd
    #commands.getoutput(cmd)

    # create white matter parcellation derived frol the cortical parcellation.
    mkwm2009 = 'mri_aparc2aseg --s %s --labelwm --hypo-as-wm --rip-unknown --volmask --o mri/wmparc2009.mgz --ctxseg aparc.a2009s+aseg.mgz --a2009s'%subjectCode
    print mkwm2009
    commands.getoutput(mkwm2009)

    # convert freesurfer file to nifti format
    cmd = 'mri_convert mri/wmparc2009.mgz /hpc/crise/hao.c/data/%s/parc_freesurfer/wmparc2009.nii' % subjectCode
    print cmd
    commands.getoutput(cmd)