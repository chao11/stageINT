# hao.c
#17/06/2016

# This is just to project the predict value onto surface
# In order to visualize(overlay) the surface, the projected file must have the same size as surface file (white or inflated)!!!!!

import commands
import os.path as op
import os

# ===================== project the predict map onto surface ===========================================================
model = 'distance'
altas_list = ['destrieux','aparcaseg', 'wmparc']
hemi = ['lh' , 'rh']
y_list = ['rspmT_0001','rspmT_0002','rspmT_0003','rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']


fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)


for subject in subjects_list:
    for parcel_altas in altas_list:
        for hemisphere in hemi:
            for y_file in y_list:

                cmd = '%s/mris_convert /hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/%s/surf/%s.inflated  ' \
                      '/hpc/crise/hao.c/data/%s/surface/%s.inflated.gii' %(fs_exec_dir ,subject,hemisphere,subject, hemisphere)
                #commands.getoutput(cmd)

                predict_subject = op.join(root_dir, subject,'predict')
                predict_output_path = op.join(predict_subject, '{}_{}_{}_predi_{}.nii.gz'.format(hemisphere, model, parcel_altas, y_file))
                if op.isfile(predict_output_path):
                    proj2surf_path = op.join(predict_subject, '{}_{}_{}_predi_{}.gii'.format(hemisphere, model, parcel_altas, y_file))
                    cmd ='{}/mri_vol2surf --src {} --regheader {} --hemi {} --o {}  --out_type gii --projfrac 0.5 --surf white'.format(fs_exec_dir, predict_output_path, subject,hemisphere, proj2surf_path)
                    print cmd
                    commands.getoutput(cmd)
                else :
                    print('no predict file')


