#! /usr/bin/python -u
# coding=utf-8

import os
import commands
import sys
import os.path as op


tracto_name = 'tracto_volume/LH_big_STS+STG_destrieux_500'

#tracto_name = str(sys.argv[1])

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list:
    subject_dir = op.join(root_dir,subject)
    #tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_{}'.format(hemisphere.upper(),tarcto_parcel))

    tracto_dir = op.join(subject_dir,'%s'%(tracto_name))
  #  cmd = 'mv /hpc/crise/hao.c/data/%s/raw_dwi /envau/work/crise/hao.c/data/%s/' %(subject,subject)

    if op.isfile('%s/fdt_matrix2.zip'%tracto_dir):
        print subject+ " done"

    elif not op.isfile('%s/fdt_matrix2.zip'%tracto_dir) and op.isfile('%s/fdt_matrix2.dot'%tracto_dir):
        cmd='zip %s/fdt_matrix2.zip %s/fdt_matrix2.dot' %(tracto_dir,tracto_dir)
        print cmd
        commands.getoutput(cmd)
        commands.getoutput('rm %s/fdt_matrix2.dot' %(tracto_dir))

    elif not op.isfile('%s/fdt_matrix2.dot'%tracto_dir):
        print '\n' + tracto_dir+' not exist yet'