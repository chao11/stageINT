#! /usr/bin/python -u


import os
import commands
import sys
import os.path as op
root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list[0:20]:
    subject_dir = op.join(root_dir,subject)
    output_tracto_dir = op.join(subject_dir,'tracto','LH_STS+STG_destrieux')
    connmatrix_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')
    if not op.isfile(connmatrix_path):
        cmd ='frioul_batch -M "[[%s]]" ../../python_scripts/co_matrix/read_probtrack_Matrix.py ' %subject
        print cmd
    else:
        print'%s connmatrix exits' %subject
