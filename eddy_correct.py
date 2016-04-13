#!
# hao.c
# 23/03/2016
# batch script for eddy correction
# loop over all the subject to launch FSL-FDT for eddy correction



import os.path as op
import os
import commands

#doc = pd.read_csv('/hpc/crise/hao.c/VOICELOC_list/completeDTIlist.csv')

root_dir = '/hpc/crise/hao.c/data'

#subjectlist = doc["Subject CODE"]

subjectlist = os.listdir(root_dir)
for subjectCode in subjectlist:
    path = "%s/%s/raw_dwi" % (root_dir,subjectCode)

    if not op.isfile("%s/data.nii.gz" % (path)):
        print("subject %s: eddy correct not existe,do it now:" % subjectCode)
        cmd = "frioul_batch 'fsl5.0-eddy_correct %s/*.nii.gz %s/data.nii.gz 0 '" % (path,path)
        print (cmd)

        commands.getoutput(cmd)
    else:
        print("subject %s: eddy correction done" % subjectCode)
