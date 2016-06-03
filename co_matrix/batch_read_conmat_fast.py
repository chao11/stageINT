#! /usr/bin/python -u
# coding=utf-8


import os
import os.path as op
import commands
import sys


hemisphere = 'lh'
tracto_parcel = 'aparcaseg'

"""
tracto_parcel = str(sys.argv[2])
hemisphere = str(sys.argv[1])
"""
root_dir = '/hpc/crise/hao.c/data'

subjects_list = os.listdir(root_dir)


for subject in subjects_list:

    subject_dir = op.join(root_dir, subject)

    tracto_dir = op.join(subject_dir, 'tracto', '{0}_STS+STG_{1}'.format(hemisphere.upper(), tracto_parcel))
    fdt_fullmatrix_path = op.join(tracto_dir, 'fdt_matrix2.zip')

    connmatrix_path = op.join(tracto_dir, 'conn_matrix_seed2parcels.jl')

    if not op.isfile(fdt_fullmatrix_path):
        print fdt_fullmatrix_path + ' not exsits'
    elif not op.isfile(connmatrix_path):
        cmd ='frioul_batch -M "[[\'%s\'], [\'%s\'],[\'%s\'] ]" /hpc/crise/hao.c/python_scripts/co_matrix/read_connmatrix_fast.py  ' %( subject, hemisphere, tracto_parcel)
        print cmd
        #commands.getoutput(cmd)