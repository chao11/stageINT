#!
# hao.c
# 29/03/2016
# loop over all the subjects for the following steps;
# 1.convert data(dcm2nii) and move to the user's workspace
# 2.eddy correction
# 3.bvec correct and change the name of bvecs and bvals files (used in bedpostX)
# 4.BET mask extraction of nodif image
# 5.bedpostX

import sujet_profiles as sp
import Convert_dcm2nii as dcm2nii
import pandas as pd
import os
import FDT_functions as FDT

"""
# convert DICOM to Nii
incompleteList = []
for subjectCode in additonal_list:

    subject = sp.Subject(subjectCode)
    dcm2nii.convert(subjectCode,subject.DTIcode)

# check if the dti data has been converted completely
    complete = dcm2nii.checkDir(subject.dwi_workspace)
    if complete == 0:
        incompleteList.append(subjectCode)

print (incompleteList)
# DICOM to NII anat et dwi en meme temps
#dcm2nii.convert(subjectCode,subject.DTIcode)
"""

# FDT processing in user_workspace: /hpc/crise/hao.c/data/subject

root_dir = '/hpc/crise/hao.c/data'

subjectlist = os.listdir(root_dir)

additonal_list = ['ACE12']

for subjectCode in additonal_list:
    path = "%s/%s/raw_dwi" % (root_dir,subjectCode)

   # FDT.eddyCorrect(subjectCode, path)
   # FDT.bvec_bval_correct(path)
   # FDT.bet(subjectCode, path)
    FDT.bedpost_batch(subjectCode,path)
    FDT.registration(subjectCode)
