
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

def registration(subjectCode):

    if len(os.listdir("/hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms" %subjectCode)) < 6:
        cmd1 = "fsl5.0-flirt -in /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/nodif_brain -ref /hpc/crise/hao.c/sanlm/%s/T1_brain.nii.gz -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2str.mat -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -cost corratio" %(subjectCode,subjectCode,subjectCode)
        cmd2 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2diff.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2str.mat" %(subjectCode,subjectCode)
        cmd3 = "fsl5.0-flirt -in /hpc/crise/hao.c/sanlm/%s/T1_brain.nii.gz -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2standard.mat -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -cost corratio" %(subjectCode,subjectCode)
        cmd4 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/standard2str.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2standard.mat" %(subjectCode,subjectCode)
        cmd5 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2standard.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2standard.mat" %(subjectCode,subjectCode)
        cmd6 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/standard2diff.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2standard.mat" %(subjectCode,subjectCode)

        cmd = "frioul_batch '%s;%s;%s;%s;%s;%s'" % (cmd1,cmd2,cmd3,cmd4,cmd5,cmd6)
        print cmd

        #commands.getoutput(cmd)
        print "%s Done!" %subjectCode
    else:
        print "%s Done!" %subjectCode

def fsRegistration(root_dir,SUBJECTS_DIR,subjectCode):

    print 'computing the registrastion matrix:'
    cmd = 'tkregister2 --mov %s/%s/mri/orig.mgz --targ %s/%s/mri/rawavg.mgz --regheader --reg junk --fslregout freesurfer2struct.mat --noedit; ' %(SUBJECTS_DIR,subjectCode,SUBJECTS_DIR,subjectCode)
    print cmd
    commands.getoutput(cmd)
    cmd2 = 'fsl5.0-convert_xfm -omat struct2freesurfer.mat -inverse freesurfer2struct.mat;'
    print cmd2
    commands.getoutput(cmd2)
    cmd3 = 'fsl5.0-flirt -in %s/%s/raw_dwi/data -ref %s/%s/sanlm/sanlm_co* -omat fa2struct.mat;' %(root_dir,subjectCode,root_dir,subjectCode)
    print cmd3
    commands.getoutput(cmd3)
    cmd4 = 'fsl5.0-convert_xfm -omat struct2fa.mat -inverse fa2struct.mat;fsl5.0-convert_xfm -omat fa2freesurfer.mat -concat struct2freesurfer.mat fa2struct.mat;fsl5.0-convert_xfm -omat freesurfer2fa.mat -inverse fa2freesurfer.mat;'
    commands.getoutput(cmd4)
    print cmd4



def probtrack(subjectCode):

    print "probtrackX"