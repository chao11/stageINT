# hao.c
# 06/06/2016
"""
Ward parcellation of connectivity matrix on surface (surfacic_connectivity_profile_STSSTG.jl)
the parcellation result is saved in the directory "surface" of the correspondence subject

"""

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

"""
hemi = 'lh'
altas = 'destrieux'
nb_clusters = 3
subject = 'AMA02'

"""
subject = str(sys.argv[1])
hemi = str(sys.argv[2])
nb_clusters = int(sys.argv[3])
altas = str(sys.argv[4])

surface = 'white'

# freesurfer setup:
fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
projection_method = '--projfrac-avg 0 1 0.1'
#projection_method = '--projfrac 0.5'
fs_subject_dir ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

root_dir = '/hpc/crise/hao.c/data'
subject_list = os.listdir(root_dir)

subject_path = op.join(root_dir,subject)
tracto_name = '{}_STS+STG_{}'.format(hemi.upper(), altas)
tracto_dir = op.join(subject_path, 'tracto', tracto_name)
surface_dir = op.join(subject_path, 'surface')

inflated_surface_path = op.join(fs_subject_dir, subject, 'surf', '{}.inflated'.format(hemi.lower()))
white_surface_path = op.join(fs_subject_dir, subject, 'surf', '{}.white'.format(hemi))
# mesh is the *h.white.gii
mesh_path = op.join(surface_dir,'{}.white.gii'.format(hemi))
# path_target = op.join(subject_path,'freesurfer_seg', 'target_mask_destrieux.nii.gz')

seedroi_gii_path = op.join(subject_path,'freesurfer_seg', '{}_STS+STG.gii'.format(hemi))

connmat_path = op.join(tracto_dir,'conn_matrix_seed2parcels.jl')
connmat_proj_path = op.join(tracto_dir, 'surfacic_connectivity_profile_STSSTG_{}.jl'.format(hemi.lower()))

output_gii_parcellation_path=op.join(surface_dir,'norma_{}.{}.parcellation_cl{}.gii'.format(hemi, altas, nb_clusters))

print 'subject ID: {}\n surface parcellation for {}, altas = {}, number of clusters = {}'.format(subject, hemi, altas, nb_clusters)

print surface_dir
# convert the mesh gifti file
if not op.isdir(surface_dir):
    os.mkdir(surface_dir)

if not op.isfile(mesh_path):
    print'*h.white.gii not exists yet, convert now.......'

    cmd = '%s/mris_convert %s  %s' %(fs_exec_dir, white_surface_path, mesh_path)
    # print cmd
    commands.getoutput(cmd)

# ===================parcellation ======================================================================================

# load connectivity matrix
surf_connmat = jl.load(connmat_proj_path)
print "surfacic connectivity matrix: {} ".format(surf_connmat.shape)

# normalize the connmat
from sklearn.preprocessing import normalize
surf_connmat = normalize(surf_connmat,norm='l2')

# read surface seed roi mask
seedroi_gii = ng.read(seedroi_gii_path)
seedroi_data = seedroi_gii.darrays[0].data # the values of seed vertex
surfmask_inds = np.flatnonzero(seedroi_data) # return the indices that are non-zeros in seedroi_data

# perform a hierarchical clustering considering spatial neighborhood (ward)
print 'ward: number of clusters:{}'.format(nb_clusters)
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

for i in range(len(np.unique(labelsf))):
    print 'label %d: %d' % (i,len(labelsf[labelsf==i]))
# write the final gifti parcellation
print 'write parcellation.gii'
ii=0
for i in surfmask_inds:
    seedroi_gii.darrays[0].data[i]=labelsf[ii]+2
    ii=ii+1


ng.write(seedroi_gii,output_gii_parcellation_path)
print 'save parcellation:\n'+output_gii_parcellation_path

# visualization of the parcellation in freeview, overlay on the inflated sueface
cmd_freeview = '%s/freeview -f %s:overlay=%s &' %(fs_exec_dir, inflated_surface_path, output_gii_parcellation_path)
print(cmd_freeview)
# commands.getoutput(cmd_freeview)  # 'freeview.bin: Command not found.'