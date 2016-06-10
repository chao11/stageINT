# hao.c
# 10/06/2016
"""
create surfacic seed mask use the connectivity matrix:
separate the connectivity matrix and create the profile image (Nifti) for each target, project the volume nto surface (mri_vol2surf)
sum all the profile and compute the seed on surface

"""
import nibabel as nib
import nibabel.gifti as ng
import numpy as np
import os
import os.path as op
import sys
import joblib as jl
import commands


subject = 'AHS22'
hemi = 'lh'
altas = 'destrieux'
#projection_method = '--projfrac-avg 0 1 0.1'
projection_method = '--projfrac 0.5'

fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
root_dir = '/hpc/crise/hao.c/data'
tracto_name = 'LH_STS+STG_destrieux'
tracto_path = op.join(root_dir,subject, 'tracto', tracto_name)
connmat_path = op.join(tracto_path,'conn_matrix_seed2parcels.jl')

coord_file_path = '/hpc/crise/hao.c/data/AHS22/tracto/LH_STS+STG_destrieux/coords_for_fdt_matrix2'
seed_nii = nib.load('/hpc/crise/hao.c/data/AHS22/freesurfer_seg/lh_STS+STG.nii.gz')
surface_dir = op.join(root_dir, subject, 'surface')

with open(coord_file_path,'r') as f:
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

mat = jl.load(connmat_path)[0]
nii = np.zeros((256,256,256))
n_target = mat.shape[1]

# compute the profile (nifti image) for each target
for t in range(n_target):
    valeur = mat[:,t]

#   compute the nifti image corresponding to this target
    for j in range(len(seed_coord)):
        c = seed_coord[j]
        nii[c[0],c[1],c[2]] = valeur[j]
#   save the nifti image
    img = nib.Nifti1Image(nii, seed_nii.get_affine())
    nii_output_path = op.join(surface_dir, '{}.nii'.format(t))
    img.to_filename(nii_output_path)

#   peoject the profil onto surface
    gii_text_output_path = op.join(surface_dir, '{}.gii'.format(t))
    proj_cmd = '%s/mri_vol2surf --src %s --o %s --out_type gii --regheader %s --hemi %s %s  ' % (fs_exec_dir, nii_output_path, gii_text_output_path, subject, hemi, projection_method )
    #commands.getoutput(proj_cmd)
    print proj_cmd


# relaod the gifti files and cumpute the seed surface mask
size = len(ng.read(gii_text_output_path).darrays[0].data)
connmat_proj = np.zeros((size, n_target))

for t in range(n_target):
    gii_text_output_path = op.join(surface_dir, '{}.gii'.format(t))
    g = ng.read(gii_text_output_path).darrays
    connmat_proj[:,t] = g[0].data
# save the projected connmat
connmat_prjo_path = op.join(tracto_path, 'connmat_proj.jl')
jl.dump(connmat_proj, connmat_prjo_path, compress=3)
print "connmat_proj saved "

# sum of theprofile for each target
sum = connmat_proj.sum(axis=1)
seedroi_gii = np.zeros((size,))
idx = np.where(sum!=0)
seedroi_gii[np.where(sum!=0)]=1

seed_gii_path = op.join(surface_dir, '{}_seed.gii'.format(hemi))
ng.write(seedroi_gii, seed_gii_path)




# compare with the projection of seed_msk and fdy_path
seed_mask_vlo2surf = '/hpc/crise/hao.c/data/AHS22/freesurfer_seg/lh_seed.gii'
g_seed = ng.read(seed_mask_vlo2surf).darrays

fdtpath_surface = '/hpc/crise/hao.c/data/AHS22/surface/surf_proj_fdtpath.gii'
fdt_gii = ng.read(fdtpath_surface).darrays