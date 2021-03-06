#!
# hao.c
# 23/03/2016
#FSL tool doesn't correct the bvecs.
# This scrip loop over the subject and correct the bvec by using a shell script.
# THen use BET extract the brain mask with threashold  = 0.35

import os
import os.path as op
import commands

# correct the bvec:
root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)


additonal_list = ["ACE12","NCA23","EMN02","MJN13" ]

for subjectCode in additonal_list:

    path = "%s/%s/raw_dwi" % (root_dir,subjectCode)

    print(path)
    print(os.listdir(path))


    # correct the bvecs file and rename as bvecs
    if not op.isfile("%s/bvecs" % (path)):
        cmd1 = 'sh /hpc/crise/hao.c/shell_scripts/ecclog2mat.sh %s/data.ecclog ' % path
        print cmd1
        commands.getoutput(cmd1)

        cmd2 ='sh /hpc/crise/hao.c/shell_scripts/rotbvecs %s/*.bvec %s/bvecs mat.list' % (path,path)
        print cmd2
        commands.getoutput(cmd2)

    # rename the bvals file for bedpostX
    if not op.isfile("%s/bvals" % (path)):
        commands.getoutput("mv %s/*.bval %s/bvals" % (path,path))


    ## BET
    if not op.isfile("%s/nodif_brain_mask" % path):
        cmdBET = 'fsl5.0-bet %s/data %s/nodif_brain  -f 0.250 -g 0 -m ' %(path,path)
        print cmdBET
        commands.getoutput(cmdBET)

    print ("%s Done" % subjectCode)