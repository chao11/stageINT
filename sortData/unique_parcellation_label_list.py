import numpy as np
import nibabel
import commands
import os

root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)


unique_list = []
for subjectCode in subjectlist[1:2]:
    parc_nii = nibabel.load('/hpc/crise/hao.c/data/%s/parc_freesurfer/wmparc.nii' %subjectCode)

    parc_data = parc_nii.get_data()
    np.unique(parc_data)
    unique_list = np.union1d(unique_list,np.unique(parc_data))


print str(unique_list)
np.savetxt("/hpc/crise/hao.c/parcel_number_list_wmparc", unique_list,fmt='%d',delimiter='\n')


