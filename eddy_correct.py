#!
# hao.c
# 23/03/2016
# batch script for eddy correction
# loop over all the subject to launch FSL-FDT for eddy correction

import os
import FDT_functions

root_dir = '/hpc/crise/hao.c/data'


subjectlist = os.listdir(root_dir)

additonal_list = ["CLH01","MON02_2_L" ]

for subjectCode in additonal_list:

    path = "%s/%s/raw_dwi" % (root_dir,subjectCode)

    FDT_functions.eddyCorrect(subjectCode, path)

