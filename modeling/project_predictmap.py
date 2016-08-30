# hao.c
# 17/06/2016

# This is used to project the predict nifti file  onto white surface
# In order to visualize(overlay) the surface, the projected file must have the same size as surface file !!!!!

import commands
import os.path as op
import os

# ===================== project the predict map onto surface ===========================================================
model = 'connmat'
altas_list = ['destrieux_WM', 'desikan_WM', 'wmparc']
hemi = ['lh', 'rh']
y_list = ['rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']
weight = 'none'   # distance or none
lateral = 'bilateral'    # ipsi or bilateral
options = ['none', 'waytotal', 'norm', 'sum', 'partarget', 'zscore']

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list[36:37]:
    for parcel_altas in altas_list[0:1]:
        for hemisphere in hemi[1:]:
            for y_file in y_list[0:1]:
                for norma in options:

                    cmd = '%s/mris_convert /hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/%s/surf/%s.inflated  ' \
                          '/hpc/crise/hao.c/data/%s/surface/%s.inflated.gii' % (fs_exec_dir, subject, hemisphere, subject, hemisphere)
                    # commands.getoutput(cmd)

                    tracto_dir = 'tracto_volume/%s_small_STS+STG_%s_5000' % (hemisphere.upper(), parcel_altas)
                    predict_subject = op.join(root_dir, subject, tracto_dir, 'predict')
                    predict_file_name = '%sWeighted_%s_%s_%s_%s_%s_predi_%s'\
                        % (weight, hemisphere, lateral, model, norma, parcel_altas, y_file)

                    predict_nii_path = op.join(predict_subject, '%s.nii.gz' % predict_file_name)
                    proj2surf_path = op.join(predict_subject, '%s.gii' % predict_file_name)

                    if op.isfile(predict_nii_path) and (not op.isfile(proj2surf_path)):

                        cmd = '%s/mri_vol2surf --src %s --regheader %s --hemi %s --o %s  --out_type gii --projfrac 0.5' \
                              ' --surf white' % (fs_exec_dir, predict_nii_path, subject, hemisphere, proj2surf_path)
                        print cmd
                        commands.getoutput(cmd)
                    elif not op.isfile(predict_nii_path):
                        print('predict file not exists %s' % predict_nii_path)


