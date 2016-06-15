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

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'

root_dir = '/hpc/crise/hao.c/data'
SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

surf='white' # white or inflated

tracto_dir = op.join('tracto', '{}_STS+STG_{}'.format(hemi.upper(), altas))
subjectList = os.listdir(root_dir)

for subject in subjectList[0:5]:
    data_path = op.join(root_dir, subject, tracto_dir, 'fdt_paths.nii.gz')
    freesurfer_mesh_reference = op.join(SUBJECTS_DIR, subject, 'mri', 'brain.mgz')
    mesh_ref = op.join(root_dir, subject, 'brain.nii.gz')

    output = op.join(root_dir, subject, 'surface','{}_surf_proj_{}_fdtpath'.format(hemi, surf))
    surface_file = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}'.format(hemi, surf))

#   convert freesurfer's brain reference for FSL:
    asc_surf_file = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.asc'.format(hemi, surf))
    convert_mgz = '%s/mris_convert %s %s' %(fs_exec_dir, freesurfer_mesh_reference, mesh_ref)
    # commands.getoutput(convert_mgz)
    print convert_mgz

#   convert the surface file to ascii filr:
    asc_surf_file = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.asc'.format(hemi, surf))
    convert_asc = '%s/mris_convert %s %s' %(fs_exec_dir, surface_file,asc_surf_file)
    # commands.getoutput(convert_asc)
    print convert_asc


    cmd = 'fsl5.0-surf_proj --data={} --surf={} --meshref={} --out={} --surfout' .format(data_path, asc_surf_file, mesh_ref, output)

    print cmd
    # commands.getoutput(cmd)