#! /usr/bin/python -u
# coding=utf-8


# HAO.C 12/04/2016
# seed parcellation /clustering using Ward's method
# loop over all the subject for parcellation, parameters defined by user
# run in batch with parameters: -M [hemisphere(lh/rh), cluster number , norm option]

# ==================================
# options for normalisation: 1. 'none': without normalisation (default)
#                            2. 'norm1': normalisation by l2
#                            3. 'standard': standardize the feature by removing the mean and scaling to unit variance



import os
import commands

import sys


hemisphere = str(sys.argv[1])
nb_cluster = int(sys.argv[2])
norma = str(sys.argv[3])

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list:
    cmd ='frioul_batch -M "[[\'%s\'], [\'%s\'],[%s],[\'%s\'] ]" /hpc/crise/hao.c/python_scripts/co_matrix/parcellation_modi.py  ' %( subject, hemisphere, str(nb_cluster), norma)
    print cmd
    # exemple cmd: frioul_batch -M "[['AHS22'],[10], ['lh'] ]" ../../python_scripts/co_matrix/multi_batch_seed_CBP_ward_parcellation.py

    commands.getoutput(cmd)
