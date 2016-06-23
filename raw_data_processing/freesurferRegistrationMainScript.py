# freesurfer registration
# called freesurfer_setup and export FSLOUTPUTTYPE=NIFTI_GZ before running this script

import shutil
import os
import commands


root_dir = '/hpc/crise/hao.c/data'
subjectList = os.listdir(root_dir)

SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

# define the FSLOUTPUTTYPE before registration
#commands.getoutput('export FSLOUTPUTTYPE=NIFTI_GZ')

for subj in subjectList:
    xfm = "/hpc/crise/hao.c/data/%s/freesurfer_regist" %subj

#   verifier qu'il y a 6 fichier .mat et un fuchier 'junk' est cree  automatiquement par tkregist
    if os.path.isdir(xfm) and len(os.listdir(xfm))==7:
        print '%s done'%subj
    else:
        shutil.rmtree(xfm)
        os.mkdir(xfm)
        os.chdir(xfm)

#       registration
        cmd = '/hpc/soft/freesurfer/freesurfer/bin/tkregister2 --mov %s/%s/mri/orig.mgz --targ %s/%s/mri/rawavg.mgz --regheader --reg junk --fslregout freesurfer2struct.mat --noedit; ' %(SUBJECTS_DIR,subj,SUBJECTS_DIR,subj)
        print cmd
        #commands.getoutput(cmd)

        cmd2 = 'fsl5.0-convert_xfm -omat struct2freesurfer.mat -inverse freesurfer2struct.mat;'
        print cmd2
        #commands.getoutput(cmd2)

        cmd3 = 'fsl5.0-flirt -in %s/%s/raw_dwi/data -ref %s/%s/sanlm/sanlm_co* -omat fa2struct.mat;' %(root_dir,subj,root_dir,subj)
        print cmd3
        #commands.getoutput(cmd3)

        cmd4 = 'fsl5.0-convert_xfm -omat struct2fa.mat -inverse fa2struct.mat;fsl5.0-convert_xfm -omat fa2freesurfer.mat -concat struct2freesurfer.mat fa2struct.mat;fsl5.0-convert_xfm -omat freesurfer2fa.mat -inverse fa2freesurfer.mat;'
       # commands.getoutput(cmd4)
        print cmd4

