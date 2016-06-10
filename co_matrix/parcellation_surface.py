# hao.c
# 06/06/2016
# Ward parcellation of connectivity matrix on surface

from sklearn.cluster import WardAgglomeration
from scipy import sparse
import nibabel.gifti as ng
from mne import spatial_tris_connectivity
import numpy as np
import os.path as op
import sys
import joblib as jl


hemi = 'lh'
altas = 'destrieux'
nb_clusters = 3

root_dir = '/hpc/crise/hao.c/data'
subject = 'AHS22'
subject_path = op.join(root_dir,subject)
tracto_name = '{}_STS+STG_{}'.format(hemi.upper(), altas)
mesh_path = op.join(subject_path,'surface', 'lh_surf_proj_inflated_fdtpath.gii')
path_tracto = op.join(subject_path, 'tracto', tracto_name)
path_target = op.join(subject_path,'freesurfer_seg', 'target_mask_destrieux.nii.gz')
path_connectivity_matrix = op.join(path_tracto,'fdt_matrix2.dot')
seedroi_gii_path = op.join(subject_path,'freesurfer_seg', '{}_seed.gii'.format(hemi))
connmat_path = op.join(path_tracto, 'conn_matrix_seed2parcels.jl')
gii_tex_parcellation_path=op.join(path_tracto,'parcellation.gii')

# load connectivity matrix
connmat = jl.load(connmat_path)[0]

# read surface seed roi mask
seedroi_gii = ng.read(seedroi_gii_path)
seedroi_data = seedroi_gii.darrays[0].data # the values of seed vertex
surfmask_inds = np.flatnonzero(seedroi_data) # return the indices that are non-zeros in seedroi_data



# perform a hierarchical clustering considering spatial neighborhood (wa
print 'ward'
g = ng.read(mesh_path)
triangles = g.darrays[1].data
# compute the spatial neighbordhood over the seed surface
connectivity = sparse.coo_matrix(sparse.csc_matrix(sparse.csr_matrix(spatial_tris_connectivity(triangles))[surfmask_inds,:])[:,surfmask_inds])

# ward clustering
ward = WardAgglomeration(n_clusters = nb_clusters, connectivity=connectivity, memory = 'nilearn_cache')
ward.fit(connmat)
labelsf = ward.labels_

# write the final gifti parcellation
print 'write parcellation.gii'
ii=0
for i in surfmask_inds:
    seedroi_gii.darrays[2].data[i]=labelsf[ii]+2
    ii=ii+1
ng.write(seedroi_gii,gii_tex_parcellation_path)