# hao.c
# 17/06/2016

# This is used to project the predict nifti file  onto white surface
# In order to visualize(overlay) the surface, the projected file must have the same size as surface file !!!!!

import commands
import os.path as op
import os

# ===================== project the predict map onto surface ===========================================================
model = 'connmat'
altas_list = ['destrieux', 'aparcaseg', 'wmparc']
hemi = ['lh', 'rh']
y_list = ['rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']
weight = 'distance'   # distance or none
lateral = 'ipsi'    # ipsi or bilateral
options = ['none', 'waytotal', 'norm', 'sum', 'partarget', 'zscore']

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list[0:2]:
    for parcel_altas in altas_list[0:1]:
        for hemisphere in hemi:
            for y_file in y_list:
                for norma in options:

                    cmd = '%s/mris_convert /hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/%s/surf/%s.inflated  ' \
                          '/hpc/crise/hao.c/data/%s/surface/%s.inflated.gii' % (fs_exec_dir, subject, hemisphere, subject, hemisphere)
                    # commands.getoutput(cmd)

                    tracto_dir = 'tracto_volume/{0}_STS+STG_{1}_2'.format(hemisphere.upper(), parcel_altas)
                    predict_subject = op.join(root_dir, subject, tracto_dir, 'predict')
                    predict_file_name = '{0}Weighted_{1}_{2}_{3}_{4}_{5}_predi_{6}'\
                        .format(weight, hemisphere, lateral, model, norma, parcel_altas, y_file)

                    predict_nii_path = op.join(predict_subject, '%s.nii.gz' % predict_file_name)
                    proj2surf_path = op.join(predict_subject, '%s.gii' % predict_file_name)

                    if op.isfile(predict_nii_path) and (not op.isfile(proj2surf_path)):

                        cmd = '{}/mri_vol2surf --src {} --regheader {} --hemi {} --o {}  --out_type gii --projfrac 0.5' \
                              ' --surf white'.format(fs_exec_dir, predict_nii_path, subject, hemisphere, proj2surf_path)
                        print cmd
                       # commands.getoutput(cmd)
                    elif not op.isfile(predict_nii_path):
                        print('predict file not exists %s' % predict_nii_path)


