#!
# hao.c
# 23/03/2016
# loop over all the subject and create bedpostX in batch mode

import os
import os.path as op
import commands

root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)
nbr = len(subjectlist)
print nbr, subjectlist[nbr-1]

for subjectCode in subjectlist:
    path = "%s/%s/raw_dwi" % (root_dir,subjectCode)
    #if not op.exists("%s/%s/raw_dwi.bedpostX" %(root_dir,subjectCode)):
    cmd = "frioul_batch 'fsl5.0-bedpostx %s/ -n 2 -b 1000'" % (path)
    print cmd
        #commands.getoutput(cmd)
#    else:
#        print("%s bedpostX done" % subjectCode)