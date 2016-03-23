#!
# hao.c
# 22/03/2016

import sujet_profiles as sp
import Convert_dcm2nii as dcm2nii
import pandas as pd

doc = pd.read_csv('/hpc/crise/hao.c/VOICELOC_list/completeDTIlist.csv')
subjectlist = doc["Subject CODE"]

#print subjectlist
#subj_nbr = len(subjectlist)
#print (subj_nbr)

incompleteList = []

for subjectCode in subjectlist:

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
subjectCode = "MGA27"
subject = sp.Subject(subjectCode)
dcm2nii.convert(subjectCode,subject.DTIcode)
"""