
import os
import commands
import sys
import os.path as op

hemisphere = 'RH'
tarcto_parcel = 'destrieux'

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list:
    subject_dir = op.join(root_dir,subject)
    tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_{}'.format(hemisphere.upper(),tarcto_parcel))
    # cmd = 'mv /hpc/crise/hao.c/data/%s/raw_dwi /envau/work/crise/hao.c/data/%s/' %(subject,subject)




    if not op.isfile('%s/fdt_matrix2.zip'%tracto_dir):
        cmd='zip %s/fdt_matrix2.zip %s/fdt_matrix2.dot' %(tracto_dir,tracto_dir)
        print cmd
        commands.getoutput(cmd)
        commands.getoutput('rm %s/fdt_matrix2.dot' %(tracto_dir))
