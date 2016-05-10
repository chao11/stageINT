#! /usr/bin/python -u
# coding=utf-8


import os
import os.path as op
import commands
import sys


hemisphere = 'lh'
tracto_parcel = 'wmparc'

root_dir = '/hpc/crise/hao.c/data'

subjects_list = os.listdir(root_dir)

#cluster = np.arange(5,22,2)

for subject in subjects_list:

    subject_dir = op.join(root_dir,subject)

    tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_{}'.format(hemisphere.upper(),tracto_parcel))
    mat_dot = op.join(tracto_dir,'fdt_matrix2.dot')
    mat_zip = op.join(tracto_dir,'fdt_matrix2.zip')
    if not op.isfile(mat_dot) and not op.isfile(mat_zip):

        cmd ='frioul_batch -M "[[\'%s\'], [\'%s\'],[\'%s\'] ]" /hpc/crise/hao.c/python_scripts/co_matrix/read_connmatrix_fast.py  ' %( subject, hemisphere,tracto_parcel)
        print cmd
        #commands.getoutput(cmd)