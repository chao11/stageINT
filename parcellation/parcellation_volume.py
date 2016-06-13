#! /usr/bin/python -u
# coding=utf-8

# HAO.C 12/04/2016
# seed parcellation /clustering using Ward's method

# run in batch with parameters: < subject, cluster number, hemisphere(lh/rh)>
# exemple cmd: frioul_batch -M "[['AHS22'],[10], ['lh'] ]" ../../python_scripts/co_matrix/parcellation_volume.py
import sys
import nibabel as nib
import joblib

import os
import os.path as op
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# ======================================================================================================================
# options for normalisation: 1. 'none': without normalisation (default)
#                            2. 'norm1': normalisation by l1
#                            3. 'norm2': normalisation by l2
#                            4. 'standard': standardize the feature by removing the mean and scaling to unit variance
#                            5. 'MinMax': MinMaxScaler, scale the features between a givin minimun and maximum, often between (0,1)
# ======================================================================================================================
root_dir = '/hpc/crise/hao.c/data'


def parcellation(subject, nb_cluster, hemisphere, norma):
    global root_dir
    print('Processing subject {} for {},cluster={}, norma: {} ...'.format(subject, hemisphere, nb_cluster, norma))


    # define directories
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(fs_seg_dir,'{}_STS+STG.nii.gz'.format(hemisphere.lower()))
    target_path = op.join(fs_seg_dir,'target_mask.nii.gz')
    output_tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemisphere.upper()))
    fdt_fullmatrix_path = op.join(output_tracto_dir,'fdt_matrix2.dot')

    output_name = subject + '_' + hemisphere.upper() + '_seed_parcellisation_cl' + str(nb_cluster) + norma +'.nii.gz'
    output_path = op.join(output_tracto_dir, output_name)
    ward_output = op.join(output_tracto_dir,'ward_cl{}_{}.jl'.format(nb_cluster, norma))


    # load connectivity matrix:
    if norma=='none':
        conn_matrix_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')
        conn_matrix = joblib.load(conn_matrix_path)
        connect = conn_matrix[0]

    else:
        conn_matrix_path = op.join(output_tracto_dir,'conn_matrix_norma_{}.jl'.format(norma))
        connect = joblib.load(conn_matrix_path)


    # check the connectivity matrix file:
    if not op.isfile(conn_matrix_path):
        print "connectivity for subject %s not exits: %s" %(subject,conn_matrix_path)


    #elif op.isfile(ward_output):
     #   print "{} exits".format(ward_outout)


    else:
        # Ward cluatering result not exits
        # ==============================================================================
        # loading the datas : seed masks, target masks and connectivity matrix fdt_matrix2.dot
        # loading the seed
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

            for j in range(shape[1]):



                for i in range(shape[0]-1,-1,-1):

                    if seed[i, j, k] > 0:

                        connect_use[i,j,k]=index

                        connect_use2[index]=[i,j,k]

                        index=index+1

        #==============================================================================
        # CLUSTERING STAGE
        # USE WARD

        # ======connectivity (constrain) matrix==========================================
        print 'compute adjacency matrix...'
        # compute the adjacency matrix over the target mask
        from sklearn.neighbors import kneighbors_graph
        connectivity = kneighbors_graph(connect_use2, 7,include_self=False)

        # ======clustering=============================================================
        print 'ward clustering...'
        #   perform a hierarchical clustering considering spatial neighborhood
        ward = AgglomerativeClustering(n_clusters = nb_cluster, linkage='ward',connectivity=connectivity)


        ward.fit(connect)
        labelsf = ward.labels_

        # save the clustering result
        joblib.dump( ward, ward_output,compress=3)
        print 'save{}'.format(ward_output)

        #==============================================================================
        #saving the parcellation to a NIFTI1 image
        if not op.isfile(output_path):
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

            file = open("{}/{}_parcellation_info.txt".format(output_tracto_dir,subject),'a+')
            file.write(output_name)

            for label in np.unique(parcellation_data):

                file.write("\npacrcel %d size: %d \n" %(label,len(parcellation_data[parcellation_data==label])))

            file.close()

        print "done: ",output_path



#parametre par défaur
nb_cluster = [2,3,4,5]
hemisphere = 'rh'
norma = 'norm2'
"""

subject = str(sys.argv[1])
hemisphere = str(sys.argv[2])
nb_cluster = int(sys.argv[3])
norma = str(sys.argv[4])
"""

subject_list = os.listdir(root_dir)

for i in subject_list:
    for n in nb_cluster[3:]:
        parcellation(i, n, hemisphere, norma)

