
import os.path as op
import os
import commands
# root_dir is the workspace
root_dir = '/hpc/crise/hao.c/data'

# Use FSL-eddycorrection, the input is DTI data, output is corrected DTI data and data.ecclog
def eddycorrect(subject_code, path):

#   define input and output
    input_data = op.join(path, '*.nii.gz')
    output_dir = op.join(path, 'data.nii.gz')
    if not op.isfile(output_dir):
        print("subject %s: eddy correct not existe,do it now:" % subject_code)
        cmd = 'fsl5.0-eddy_correct %s %s 0 ' % (input_data, output_dir)
        print (cmd)
      # commands.getoutput(cmd)

        print ("Eddy correction finished! ")
    else:
        print("subject %s: eddy correction done" % subject_code)

    return


def bet(subject_code, path):

   if not op.isfile("%s/nodif_brain_mask" % path):
        cmdBET = 'fsl5.0-bet %s/data %s/nodif_brain  -f 0.250 -g 0 -m ' %(path, path)
        print cmdBET
       # commands.getoutput(cmdBET)
        print ("%s Done" % subject_code)


# This function correct the .bvec and .bvec files.
# .bval and .bval are created when convert DiCOM data to NIFTI (dcm2nii)
# correct and rename these files to 'bvecs', 'bvals' for further usage.
# the correction of .bvec is not necessary, if you want o do it, you need data.ecclog as input
def bvec_bval_correct(path):

    # data.ecclog is the output of eddy correction
    data_ecclog = op.join(path, 'data.ecclog')
    bvec_initial = op.join(path,'*.bvec')

#   out_put name of the corrected bvecs file
    bvec_corrected = op.join(path, 'bvecs')

    if not op.isfile(bvec_corrected):
        cmd1 = 'sh /hpc/crise/hao.c/shell_scripts/ecclog2mat.sh %s ' % data_ecclog
        print cmd1
      #  commands.getoutput(cmd1)

        cmd2 ='sh /hpc/crise/hao.c/shell_scripts/rotbvecs %s %s mat.list' % (bvec_initial,bvec_corrected)
        print cmd2
      #  commands.getoutput(cmd2)

    # rename the bvals file for bedpostX
    #if not op.isfile("%s/bvals" % (path)):
        #commands.getoutput("mv %s/*.bval %s/bvals" % (path,path))


def bedpost_batch(subjectCode,path):
    bedpost_dir = op.join(root_dir, subjectCode, 'raw_dwi.bedpostX')
    if not op.exists(bedpost_dir):
        cmd = "frioul_batch 'fsl5.0-bedpostx %s/ -n 2 -b 1000'" % (path)
        print cmd
        #commands.getoutput(cmd)
    else:
        print("%s bedpostX done" % subjectCode)


def registration(subjectCode):
    xfms_path = op.join(root_dir, subjectCode,'raw_dwi.bedpostX', 'xfms')

    if len(os.listdir(xfms_path)) < 6:
        cmd1 = "fsl5.0-flirt -in /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/nodif_brain -ref /hpc/crise/hao.c/sanlm/%s/T1_brain.nii.gz -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2str.mat -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -cost corratio" %(subjectCode,subjectCode,subjectCode)
        cmd2 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2diff.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2str.mat" %(subjectCode,subjectCode)
        cmd3 = "fsl5.0-flirt -in /hpc/crise/hao.c/sanlm/%s/T1_brain.nii.gz -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2standard.mat -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -cost corratio" %(subjectCode,subjectCode)
        cmd4 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/standard2str.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2standard.mat" %(subjectCode,subjectCode)
        cmd5 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2standard.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/str2standard.mat" %(subjectCode,subjectCode)
        cmd6 = "fsl5.0-convert_xfm -omat /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/standard2diff.mat -inverse /hpc/crise/hao.c/data/%s/raw_dwi.bedpostX/xfms/diff2standard.mat" %(subjectCode,subjectCode)

        cmd = "frioul_batch '%s;%s;%s;%s;%s;%s'" % (cmd1,cmd2,cmd3,cmd4,cmd5,cmd6)
        print cmd

        #commands.getoutput(cmd)
        print "%s Done!" % subjectCode
    else:
        print "%s Done!" % subjectCode

