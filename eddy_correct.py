#!
# hao.c
# 23/03/2016
# batch script for eddy correction

import pandas as pd
import os.path as op
import os
import commands

doc = pd.read_csv('/hpc/crise/hao.c/VOICELOC_list/completeDTIlist.csv')

root_dir = '/hpc/crise/hao.c/data'

subjectlist = doc["Subject CODE"]


for subjectCode in subjectlist:
    path = '{}/{}/raw_dwi'.format(root_dir,subjectCode)

    if op.isfile("{}/data.nii.gz".format(path))==False:
        print("subject %s: eddy correct not existe,do it now:" %subjectCode)
        cmd = "frioul_batch 'fsl5.0-eddy_correct {}/*.nii.gz {}/data.nii.gz 0 '".format(path,path)
        print (cmd)

        commands.getoutput(cmd)
    else:
        print("subject %s: eddy correction done" % subjectCode)
