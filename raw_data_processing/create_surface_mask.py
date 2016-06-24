#! /usr/bin/python -u
# coding=utf-8

"""
hao.c
22/06/2016

create surfacic seed mask and target mask for tractography
seed option: big/small seed
target option: altas

by default is 'aparc' : get annotation from  $SUBJECTS_DIR/label/*h.aparc.annot
use annot.a2009s.annot to create the small seed region, aparc for big seed region

ex parameters:
   - altac = 'destrieux'
   - annot = 'aparc.a2009s'
   - hemisphere = 'rh'
"""
import commands
import os
import  os.path as op
import sys

hemisphere = str(sys.argv[1])
seed = str(sys.argv[2])

if seed == 'big':
    annot = 'aparc'

elif seed == 'small':
    annot = 'aparc.a2009s'


fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
root_dir = '/hpc/crise/hao.c/data'
#subject = 'AHS22'

subject_list = os.listdir(root_dir)

for subject in subject_list:
    label_basedir = op.join(root_dir, subject, 'label')
    label_outdir = op.join(label_basedir, annot)
    path = op.join(label_outdir,hemisphere)
    surface_dir = op.join(root_dir, subject, 'surface')
    white_surface_file = op.join(surface_dir, '%s.white.gii'%hemisphere )

    listOfseed = op.join(label_basedir, 'listOf_%s_STS+STG_%s.txt'%(seed, hemisphere))
    output_seed = op.join(root_dir, subject, 'freesurfer_seg', '%s_%s_STS+STG.gii'%(hemisphere, seed))

    # check if the base dir of label exstis:
    if not op.isdir(label_basedir):
        print(label_basedir + 'not exists, create the directory...')
        os.mkdir(label_basedir)

    # 1. convert an annotation file into multiple label files.
    cmd1 = '%s/mri_annotation2label --subject %s --hemi %s --annotation %s --outdir %s' % (fs_exec_dir, subject, hemisphere, annot, label_outdir)
    print cmd1
    commands.getoutput(cmd1)

    # 2. create the list of labels which you want to extract ( list of .label files )
    if annot == 'aparc':
        cmd2 = 'echo "%s.bankssts.label\n%s.middletemporal.label\n%s.superiortemporal.label\n%s.transversetemporal.label" > %s' %(path, path, path, path, listOfseed)

    elif annot == 'aparc.a2009s':
        cmd2 = 'echo "%s.S_temporal_sup.label\n%s.G_temp_sup-Lateral.label" > %s'%(path, path, listOfseed)
    print cmd2
    commands.getoutput(cmd2)


    # 3. create the suaface using '*h.white.gii' surface file:
    if not op.isfile(white_surface_file):
        fs_surface_dir = '/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/%s/surf/' %subject
        cmd = '%s/mris_convert %s/lh.white %s' %(fs_exec_dir, fs_surface_dir, white_surface_file)
        commands.getoutput(cmd)

    cmd3 = 'fsl5.0-label2surf -s %s -o %s -l %s' %(white_surface_file, output_seed, listOfseed)
    print(cmd3)
    commands.getoutput(cmd3)



