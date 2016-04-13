# HAO.C 12/04/2016
# seed parcellation/clustering using Ward's method

import sys
import nibabel as nib
import commands

import os
import os.path as op
import numpy as np
from sklearn.cluster import WardAgglomeration
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from scipy import sparse



# nb_cluster = int(sys.argv[4])
nb_cluster = 2

#nom de la tracto : CC_RH, M1_LH, Thalamus_RH, etc
# name_tracto = sys.argv[6]
name_tracto = 'tarcto'

# nom du conteneur : Tracto dans mon cas
tracto_basename = sys.argv[5]

# seed/target name : FROM ROOT SEGMENTATION PATH (so add Right_H/Left_H if needed)
seed_name = 'seed.nii.gz'

#path of the subject (patient/controls)
subject_path = sys.argv[7]


root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)

for subject in subjects_list:
#for ind, arg in enumerate(sys.argv[1:]):
#    subject = arg

    # define directories
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(fs_seg_dir,'lh_STS+STG.nii.gz')
    target_path = op.join(fs_seg_dir,'target_mask.nii.gz')
    output_tracto_dir = op.join(subject_dir,'tracto','RH_STS+STG_destrieux')
    fdt_fullmatrix_path = op.join(output_tracto_dir,'fdt_matrix2.dot')
    output_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')

    #==============================================================================
    #loading the datas : seed masks, target masks and connectivity matrix fdt_matrix2.dot
    #loading the seed
    seedH = nib.load(seed_path)
    #
    seed = seedH.get_data()




    #==============================================================================
    # CLUSTERING STAGE
    # USE WARD
    # sometime there are nan in the matrix after cross-correlation due to full zero line in the
    # former matrix. Hence, one replace the non-zeros with almost zero
    connect = np.nan_to_num(connect)

    #   loading the mask, and binarise

    mask = seed.astype(np.bool)
    #
    shape = mask.shape
    #
    print 'compute adjacency matrix...'
    # compute the adjacency matrix over the target mask
    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(connect_use2,7)

    print 'ward clustering...'
    #   perform a hierarchical clustering considering spatial neighborhood
    ward = WardAgglomeration(n_clusters = nb_cluster, connectivity=connectivity)
    ward.fit(np.transpose(connect))
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

    if option_percent!=1:
        connectivity_matrix_file.close()