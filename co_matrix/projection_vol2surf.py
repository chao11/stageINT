# hao.c
# 28/04/2016
# this script is used to project the volume parcellation on a surface, here we focus on the parcellation of cl=7,normalised by norm2.
# the output is the texture in GIfTI format(gii) result can be displayed in anatomist by fusion with the white or inflatesd surface(created by freesurfer)

import os
import os.path as op
import commands


hemi = 'lh'
parcel_name = 'cl7none'


root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list[2:]:

    subject_dir = op.join(root_dir,subject)
    tarcto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemi.upper()))
    src_volume = op.join(tarcto_dir,'{}_{}_seed_parcellisation_{}.nii.gz'.format(subject,hemi.upper(),parcel_name))

    parcell_dir = op.join(subject_dir,'parcellation',hemi.upper())
    output_path = '{}/{}.{}.gii'.format(parcell_dir,hemi.lower(),parcel_name)

    # project the volume on a surface
    cmd ='mri_vol2surf --src {} --regheader {} --hemi {} --o {}  --out_type gii --projfrac 0.5'.format(src_volume,subject,hemi,output_path)
    print cmd
    #commands.getoutput(cmd)
"""
        cmd2='mris_convert /hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/{}/surf/{}.inflated {}/{}.inflated.gii'.format(subject,hemi,parcell_dir,hemi)
        cmd3='mris_convert /hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db/{}/surf/{}.white {}/{}.white.gii'.format(subject,hemi,parcell_dir,hemi)
        print cmd2
        print cmd3
        commands.getoutput(cmd2)
        commands.getoutput(cmd3)
"""