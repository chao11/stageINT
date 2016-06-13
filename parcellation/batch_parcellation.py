#! /usr/bin/python -u
# coding=utf-8


# HAO.C 12/04/2016
# seed parcellation /clustering using Ward's method
# loop over all the subject for parcellation, parameters defined by user
# run in batch with parameters: -M [hemisphere(lh/rh), cluster number , norm option]

# ==================================
# options for normalisation: 1. 'none': without normalisation (default)
#                            2. 'norm2': normalisation by l2
#                            3. 'standard': standardize the feature by removing the mean and scaling to unit variance



import os
import commands
import sys


hemisphere = str(sys.argv[1])
nb_cluster = int(sys.argv[2])

norma = 'norm2'

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

#cluster = np.arange(5,22,2)

for subject in subjects_list:
   # for nb_cluster in cluster:

    cmd ='frioul_batch -M "[[\'%s\'], [\'%s\'],[%s],[\'%s\'] ]" /hpc/crise/hao.c/python_scripts/co_matrix/parcellation_volume.py  ' %( subject, hemisphere, str(nb_cluster), norma)
    print cmd

    #commands.getoutput(cmd)
