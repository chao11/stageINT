
import os.path as op
import os
import commands


def eddyCorrect(subjectCode,path):

    if not op.isfile("%s/data.nii.gz" % (path)):
        print("subject %s: eddy correct not existe,do it now:" % subjectCode)
        cmd = 'fsl5.0-eddy_correct %s/*.nii.gz %s/data.nii.gz 0 ' % (path,path)
        print (cmd)

     #   commands.getoutput(cmd)
    else:
        print("subject %s: eddy correction done" % subjectCode)

    return

def bet(subjectCode,path):

   if not op.isfile("%s/nodif_brain_mask" % path):
        cmdBET = 'fsl5.0-bet %s/data %s/nodif_brain  -f 0.250 -g 0 -m ' %(path,path)
        print cmdBET
        commands.getoutput(cmdBET)
        print ("%s Done" % subjectCode)

def bvec_bval_correct(path):
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

def bedpost_batch(subjectCode,path):

    if not op.exists("/hpc/crise/hao.c/data/%s/raw_dwi.bedpostX" %(subjectCode)):
        cmd = "frioul_batch 'fsl5.0-bedpostx %s/'" % (path)
        print cmd
        #commands.getoutput(cmd)
    else:
        print("%s bedpostX done" % subjectCode)

