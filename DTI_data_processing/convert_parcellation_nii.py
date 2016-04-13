import numpy as np
import nibabel
import commands
import os

root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)
os.chdir('/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db')


for subjectCode in subjectlist:

    os.chdir('/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/%s/mri' %subjectCode)
    cmd = 'mri_convert aparc.a2009s+aseg.mgz /hpc/crise/hao.c/data/%s/parc_freesurfer/parcellisation_freesurfer.nii' % subjectCode
    print cmd
    commands.getoutput(cmd)

"""
parc_nii = nibabel.load('parcellisation_fs.nii')
parc_data = parc_nii.get_data()

np.unique(parc_data)

"""

