
import numpy as np
import nibabel
import os
import commands


def read_coord(file_path):
    with open(file_path,'r') as f:
        file = f.read()
        text = file.split('\n')
        # print("total number of voxels:", len(text)-1)
        seed_coord = np.zeros((len(text)-1, 3), dtype= int)
        for i in range(len(text)-1):
        #
            data = np.fromstring(text[i], dtype=int, sep=" ")
        #
            seed_coord[i, 0] = data[0]
        #
            seed_coord[i, 1] = data[1]
        #
            seed_coord[i, 2] = data[2]
    return seed_coord


# =============================== main =================================================================================

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'

subject = 'AHS22'
contrast = 'rcon_0004'
hemisphere = 'rh'
altas = 'destrieux'
tracto_name = 'tracto_volume/'

coord_path = '/hpc/crise/hao.c/data/{}/tracto_volume/{}_STS+STG_{}_2/coords_for_fdt_matrix2'.format(subject, hemisphere.upper(),altas)
y_path = '/hpc/banco/voiceloc_full_database/func_voiceloc/{}/nomask_singletrialbetas_spm12_stats/resampled_fs5.3_space/{}.nii'.format(subject, contrast)
y_mask_nii_path = '/hpc/crise/hao.c/data/{}/{}_{}_mask.nii.gz'.format(subject, hemisphere, contrast)
y_mask_gii_path = '/hpc/crise/hao.c/data/{}/{}_{}_mask.gii'.format(subject, hemisphere, contrast)


coord = read_coord(coord_path)
y_data = nibabel.load(y_path).get_data()
img = np.zeros((256,256,256))
for i in coord:
    img[i[0], i[1], i[2]] = y_data[i[0], i[1], i[2]]

y_img = nibabel.Nifti1Image(img, nibabel.load(y_path).get_affine())
y_img.to_filename(y_mask_nii_path)
print 'save' + y_mask_nii_path

cmd ='{}/mri_vol2surf --src {} --regheader {} --hemi {} --o {}  --out_type gii --projfrac 0.5 --surf white'.format(fs_exec_dir, y_mask_nii_path, subject, hemisphere, y_mask_gii_path)
print cmd
commands.getoutput(cmd)
