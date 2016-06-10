# hao.c
# 08/06/2016

"""
project the connectivity matrix onto a freesurfer sueface: fsl5.0-surf_proj
output is a surface file (option --surfout )

setup freesurfer before launching the scripts

"""

import os
import os.path as op
import commands


hemi = 'lh'
altas = 'destrieux'

root_dir = '/hpc/crise/hao.c/data'
SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

tracto_dir = op.join('tracto', '{}_STS+STG_{}'.format(hemi.upper(), altas))
subjectList = os.listdir(root_dir)

for subject in subjectList[0:5]:
    data_path = op.join(root_dir, subject, tracto_dir, 'fdt_paths.nii.gz')
    surface_file = op.join(SUBJECTS_DIR, subject, 'surf', '{}.white'.format(hemi))
    asc_surf_sile = op.join(SUBJECTS_DIR, subject, 'surf', '{}.inflated.asc'.format(hemi))
    freesurfer_mesh_reference = op.join(SUBJECTS_DIR, subject, 'mri', 'brain.mgz')
    mesh_ref = op.join(root_dir, subject, 'brain.nii.gz')

    output = op.join(root_dir, subject, 'surface','{}_surf_proj_inflated_fdtpath'.format(hemi))

    cmd = 'fsl5.0-surf_proj --data={} --surf={} --meshref={} --out={} --surfout' .format(data_path, asc_surf_sile, mesh_ref, output)

    print cmd
    # commands.getoutput(cmd)