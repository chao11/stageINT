# HAO.C 12/04/2016
# seed parcellation/clustering using Ward's method

import sys
import nibabel as nib
import joblib
import commands

import os
import os.path as op
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from scipy import sparse



# nb_cluster = int(sys.argv[4])
nb_cluster = 2

root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list[0:1]:
#    subject = arg
     # define directories
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(fs_seg_dir,'lh_STS+STG.nii.gz')
    target_path = op.join(fs_seg_dir,'target_mask.nii.gz')
    output_tracto_dir = op.join(subject_dir,'tracto','LH_STS+STG_destrieux')
    fdt_fullmatrix_path = op.join(output_tracto_dir,'fdt_matrix2.dot')
    conn_matrix_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')

    output_name = 'parcellisation_ward_2cluster.nii.gz'
    output_path = op.join(output_tracto_dir,output_name)

    #==============================================================================
    #loading the datas : seed masks, target masks and connectivity matrix fdt_matrix2.dot
    #loading the seed
    seedH = nib.load(seed_path)
    seed = seedH.get_data()
    mask = seed.astype(np.bool)
    shape = mask.shape

    # m1 : how many non-zeros in seed (could have used numpy nonzero function...)
    m1 = sum(sum(sum(1 for i in row if i > 0) for row in row2) for row2 in seed)
    # index of a voxel given its position X,Y,Z
    connect_use = np.zeros((shape[0],shape[1],shape[2]))
    # position of avoxel X,Y,Z given its index
    connect_use2 = np.zeros((m1,3))
    #
    index=0
    #
    # ne pas utiliser NP.WHERE OU AUTRE nonzero
    for k in range(shape[2]):
    #
        for j in range(shape[1]):
    #
            for i in range(shape[0]-1,-1,-1):
    #
                if seed[i, j, k] > 0:
    #
                    connect_use[i,j,k]=index
    #
                    connect_use2[index]=[i,j,k]
    #
                    index=index+1
    print 'connect_use2:',connect_use2.shape
    #==============================================================================
    # CLUSTERING STAGE
    # USE WARD
    conn_matrix_seed2parcels = joblib.load(conn_matrix_path)
    connect = conn_matrix_seed2parcels[0]
    print 'connectivity matrix seed2targets:',connect.shape

    print 'compute adjacency matrix...'
    # compute the adjacency matrix over the target mask
    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(connect, 7,include_self=False)
    print 'data samples connectivity matrix :',connectivity.shape

    print 'ward clustering...'
    #   perform a hierarchical clustering considering spatial neighborhood
    ward = AgglomerativeClustering(n_clusters = nb_cluster, linkage='ward',connectivity=connectivity)
    ward.fit(connect)
    labelsf = ward.labels_
# the labels are the final labels of each voxels

    #==============================================================================
    #saving the parcellation to a NIFTI1 image
    print 'saving parcellation as a NIFTI file...'
    sh = seed.shape
    parcellation_data = np.zeros((sh[0], sh[1], sh[2]))

    index_voxel = 0

    # for each seed voxel : label the voxel
    # modify the matrix data from get_data and save it as a NIFTI nii.gz
    # a changer enutilisant np.where avec connectu_use2
    for k in range(sh[2]):
        for j in range(sh[1]):
            for i in range(sh[0]-1, -1, -1):
                if seed[i, j, k] > 0:
                    parcellation_data[i, j, k] = labelsf[index_voxel] + 2
                    index_voxel = index_voxel + 1

    sFinal=nib.Nifti1Image(parcellation_data, seedH.get_affine())
    nib.save(sFinal, output_path)

