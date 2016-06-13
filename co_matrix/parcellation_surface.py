# hao.c
# 06/06/2016
# Ward parcellation of connectivity matrix on surface

from sklearn.cluster import WardAgglomeration
from sklearn.cluster import AgglomerativeClustering

from scipy import sparse
import nibabel.gifti as ng
import nibabel as nib
from mne import spatial_tris_connectivity
import numpy as np
import os
import os.path as op
import sys
import joblib as jl
import commands


hemi = 'lh'
altas = 'destrieux'
nb_clusters = 3

# freesurfer setup:
fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
# projection_method = '--projfrac-avg 0 1 0.1'
projection_method = '--projfrac 0.5'


root_dir = '/hpc/crise/hao.c/data'
subject = 'AHS22'
subject_path = op.join(root_dir,subject)
tracto_name = '{}_STS+STG_{}'.format(hemi.upper(), altas)
tracto_dir = op.join(subject_path, 'tracto', tracto_name)
surface_dir = op.join(root_dir, subject, 'surface')
parcellation_path=op.join(tracto_dir,'parcel_surface')

mesh_path = op.join(subject_path,'surface', 'lh_surf_proj_inflated_fdtpath.gii')
# path_target = op.join(subject_path,'freesurfer_seg', 'target_mask_destrieux.nii.gz')

seegroi_nii_path = op.join(subject_path,'freesurfer_seg', '{}_STS+STG.nii'.format(hemi))
seedroi_gii_path = op.join(subject_path,'freesurfer_seg', '{}_seed.gii'.format(hemi))

surfacic_connmat_path = op.join(tracto_dir, 'surfacic_connectivity_profile_STSSTG_{}.jl'.format(hemi.lower()))
gii_tex_parcellation_path=op.join(parcellation_path,'parcellation.gii')

connmat_path = op.join(tracto_dir,'conn_matrix_seed2parcels.jl')
connmat_proj_path = op.join(tracto_dir, 'surfacic_connectivity_profile_STSSTG_{}.jl'.format(hemi.lower()))
coord_file_path = op.join(tracto_dir, 'coords_for_fdt_matrix2')

seed_nii = nib.load(seegroi_nii_path)

# ================== compute surfacic connectivity matrix===============================================================
print 'compute surfacic connectivity matrix......'

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

size = len(ng.read(seedroi_gii_path).darrays[0].data)
g_data = np.zeros((size, n_target))

# compute the profile (nifti image) for each target
for t in range(n_target):
    valeur = mat[:,t]

#   compute the nifti image corresponding to this target
    for j in range(len(seed_coord)):
        c = seed_coord[j]
        nii[c[0],c[1],c[2]] = valeur[j]
#   save the nifti image
    img = nib.Nifti1Image(nii, seed_nii.get_affine())
    nii_output_path = op.join(surface_dir, '%03d.nii'%(t+1))
    img.to_filename(nii_output_path)

#   peoject the profil onto surface
    gii_text_output_path = op.join(surface_dir, '{}.gii'.format(t))
    proj_cmd = '%s/mri_vol2surf --src %s --o %s --out_type gii --regheader %s --hemi %s %s  ' % (fs_exec_dir, nii_output_path, gii_text_output_path, subject, hemi, projection_method )
    commands.getoutput(proj_cmd)
    #print proj_cmd

#   reload gifti file
    g = ng.read(gii_text_output_path).darrays
    g_data[:,t] = g[0].data

#   remove gifti and nifti file
    os.remove(nii_output_path)
    os.remove(gii_text_output_path)

print 'shape of gift data: {} '.format(g_data.shape)

# extract the profile of the seed region and compute the profile project on the surface
sum = g_data.sum(axis=1)
seedroi_gii = np.zeros((size,))
idx = np.where(sum!=0)[0]
seedroi_gii[idx[0]]=1

connmat_proj = g_data[idx,:]

print("shape of the surfacic connmat :{}".format(connmat_proj.shape))

# save the projected connmat
jl.dump(connmat_proj, connmat_proj_path, compress=3)
print "surfacic_connectivity_profile saved \n {} \n".format(connmat_proj_path)


# ===================parcellation ======================================================================================


# load connectivity matrix
surf_connmat = jl.load(surfacic_connmat_path)
print "surfacic connectivity matrix: {} ".format(surf_connmat.shape)

# read surface seed roi mask
seedroi_gii = ng.read(seedroi_gii_path)
seedroi_data = seedroi_gii.darrays[0].data # the values of seed vertex
surfmask_inds = np.flatnonzero(seedroi_data) # return the indices that are non-zeros in seedroi_data

# perform a hierarchical clustering considering spatial neighborhood (ward)
print 'ward:......'
g = ng.read(mesh_path)
triangles = g.darrays[1].data

# compute the spatial neighbordhood over the seed surface
adjacency = sparse.coo_matrix(sparse.csc_matrix(sparse.csr_matrix(spatial_tris_connectivity(triangles))[surfmask_inds,:])[:,surfmask_inds])
print " adjacency matrix: {}".format(adjacency.shape)

# ward clustering
# ward = WardAgglomeration(n_clusters = nb_clusters, connectivity=connectivity, memory = 'nilearn_cache')
ward = AgglomerativeClustering(n_clusters = nb_clusters, linkage='ward',connectivity=adjacency)
ward.fit(surf_connmat)
labelsf = ward.labels_

# write the final gifti parcellation
print 'write parcellation.gii'
ii=0
for i in surfmask_inds:
    seedroi_gii.darrays[0].data[i]=labelsf[ii]+2
    ii=ii+1

# save the parcellation:
if not op.isdir(parcellation_path):
    os.mkdir(parcellation_path)

ng.write(seedroi_gii,gii_tex_parcellation_path)
print 'save parcellation:\n'+gii_tex_parcellation_path