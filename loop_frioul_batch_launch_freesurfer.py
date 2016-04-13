# -*- coding: utf-8 -*-
"""
Loop over all subjects to launch freesurfer5.3

@author: takerkart
"""


import commands
import os
import os.path as op
import glob

# Chang, change this to where your data is
root_dir = '/hpc/crise/hao.c/sanlm'

# Chang, change this to your list of subjects!
#subjects_list= ['ACE12']
subjects_list = os.listdir(root_dir)

for subject in subjects_list:
    # Chang, change the following so that unimported_path is the filename of the sanlm denoised file for this subject
    # unimported_path = glob.glob(op.join(root_dir,subject,'sanlm_*.nii'))[0]
    unimported_path = '%s/%s/sanlm_*.nii' %(root_dir,subject)

    # this is the freesurfer command; Chang, no need to change it!
    cmd = "frioul_batch 'freesurfer_setup; recon-all -all -3T -qcache -i %s -s %s'" % (unimported_path, subject)
    print(cmd)
    # Chang, uncomment the following line once you've launched one of the above commands in an interactive session with success
    #a = commands.getoutput(cmd)
