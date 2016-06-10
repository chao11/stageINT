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

root_dir = '/hpc/crise/hao.c/data'
subject = 'AHS22'
tracto_name = 'LH_STS+STG_destrieux'

subject_path = op.join(root_dir,subject)

mesh_path = op.join(subject_path,sys.argv[6])

path_tracto = op.join(subject_path,tracto_name)
path_target = op.join(path_tracto,'targets.nii.gz')
path_connectivity_matrix = op.join(path_tracto,'fdt_matrix2sub.dot')

gii_tex_parcellation_path=op.join(path_tracto,'parcellation.gii')
connectivity_matrix_file = open(path_connectivity_matrix,"r")
connectivity_matrix = connectivity_matrix_file.read()

seedroi_gii_path = op.join(path_tracto,sys.argv[7])


# read surface seed roi mask

seedroi_gii = ng.read(seedroi_gii_path)
seedroi_data = seedroi_gii.darrays[2].data
surfmask_inds = np.flatnonzero(seedroi_data)


# perform a hierarchical clustering considering spatial neighborhood (wa
print 'ward'
g = ng.read(mesh_path)
triangles = g.darrays[1].data
# compute the spatial neighbordhood over the seed surface
connectivity = sparse.coo_matrix(sparse.csc_matrix(sparse.csr_matrix(spatial_tris_connectivity(triangles))[surfmask_inds,:])[:,surfmask_inds])
surf_connect_mat[np.where(np.isinf(surf_connect_mat))]=0
surf_connect_mat = np.nan_to_num(surf_connect_mat)
# normalize the connectivity matrix
#min_max_scaler = pr.MinMaxScaler()
#connect = np.transpose(min_max_scaler.fit_transform(np.transpose(surf_connect_mat)))
for i in range(surf_connect_mat.shape[0]):
    surf_connect_mat[i,:] = surf_connect_mat[i,:]/surf_connect_mat[i,:].max()

# remove NaN and Inf
surf_connect_mat = np.nan_to_num(surf_connect_mat)
surf_connect_mat[np.where(np.isinf(surf_connect_mat))]=0
# ward clustering
ward = WardAgglomeration(n_clusters = nb_clusters, connectivity=connectivity, memory = 'nilearn_cache')
ward.fit(np.transpose(surf_connect_mat))
labelsf = ward.labels_

# write the final gifti parcellation
print 'write parcellation.gii'
ii=0
for i in surfmask_inds:
    seedroi_gii.darrays[2].data[i]=labelsf[ii]+2
    ii=ii+1
ng.write(seedroi_gii,gii_tex_parcellation_path)