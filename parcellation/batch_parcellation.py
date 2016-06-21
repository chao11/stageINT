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
import os.path as op
import commands
import sys

"""
hemisphere = str(sys.argv[1])
nb_cluster = int(sys.argv[2])
altas = str(sys.argv[3])
"""

hemi = 'lh'
altas = 'destrieux'
nb_clusters = 5

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

#cluster = np.arange(5,22,2)

for subject in subjects_list:
   # for nb_cluster in cluster:

    surface_dir = op.join(root_dir, subject, 'surface')
    output_gii_parcellation_path=op.join(surface_dir,'norma_{}.{}.parcellation_cl{}.gii'.format(hemi, altas, nb_clusters))

    if not op.isfile(output_gii_parcellation_path):
       # cmd ='frioul_batch -M "[[\'%s\'], [\'%s\'],[%s],[\'%s\'] ]" /hpc/crise/hao.c/python_scripts/parcellation/parcellation_surface.py  ' %( subject, hemisphere, str(nb_cluster), altas)
        cmd = 'python /hpc/crise/hao.c/python_scripts/parcellation/parcellation_surface.py {} {} {} {} '.format(subject, hemi, nb_clusters, altas)
        print cmd

       # commands.getoutput(cmd)
