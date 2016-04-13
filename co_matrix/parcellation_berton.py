#!/usr/bin/python
#==============================================================================
#
#
#    HOME  MADE  PARCELLATION (@ INT, France)
#
#
#==============================================================================
# pylint: disable=E1101,C0301,C0103, W0611
#import nilearn (needs numpy 2.xx)
#import pysparse.sparse
# %run parcellation.py 1 0 0 1 0 8 freesurfer_seg Tracto RH_M1 Right_H/RH_PreCentral /riou/work/comco/Berton/Controles/c3
import sys
import nibabel as nib
import commands

import os.path as op
import numpy as np
from sklearn.cluster import WardAgglomeration
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from scipy import sparse 

print 'loading...'

option_cluster=int(sys.argv[1])
#==============================================================================
# USER SETTABLE PARAMETERS
# option_cluster :  1=ward clustering (RECOMMENDED)
#                   2=k-means
#                   3=spectral clustering
#                   4=FSL-like clustering (hard segmentation)
#==============================================================================
# to perform a segmentation based on the freesurfer segmentation
option_percent = int(sys.argv[2])

option_print_matrix=int(sys.argv[3])
#intelligent subsampling

# number of cluster to parcellate the seed
nb_cluster = int(sys.argv[4])
#nom de la tracto : CC_RH, M1_LH, Thalamus_RH, etc
name_tracto = sys.argv[6]
# nom du conteneur : Tracto dans mon cas
tracto_basename = sys.argv[5]
# seed/target name : FROM ROOT SEGMENTATION PATH (so add Right_H/Left_H if needed)
seed_name = 'seed.nii.gz'
#path of the subject (patient/controls)
subject_path = sys.argv[7]
#==============================================================================
# mettre a 0 pour le moement (option irmf pas stable)
option_irmf=0
# utiliser la parametrisation du corps calleux dans la regularisation spatiale ?
# pour une autre ROI que le cc mettre : 0
use_custom_coordinates=0
# pondErer la taille des tractes 
option_length_tract=0
option_weight = 1
# 0 : no ponderation
# 1 : treshold : remove the small connections (according to distance_treshold)
distance_treshold = 5
# 2 : weight the matrix (don't use it now since this is not sure it makes sense...)
# with weight alpha
alpha=1
# remove the tracts that are below this limit
# treshold level when printing the mean connectivity pattern of each parcel
treshold=0
# regularisation weight
alpha2 = float(sys.argv[8])
# Tracto dans mon cas
tracto_name = op.join(tracto_basename,name_tracto)
# fichier de correspondance entre index des voxels et leur label dans le subsampling
coord_matrix = op.join(subject_path,tracto_name,'coords_subsampling')
#desired output directory
output_name = 'parcellisation_'+str(option_percent)+str(nb_cluster)+str(option_cluster)+str(alpha2)+'.nii.gz'
#
output_path = op.join(subject_path,tracto_name,output_name)
#
path_target = op.join(subject_path,tracto_name,'targets.nii.gz')
#
path_tracto = op.join(subject_path,tracto_name)
#requires the seed mask
path_seed = op.join(path_tracto,seed_name)
#requires fdt_matrix2.dot
path_connectivity_matrix = op.join(subject_path,tracto_name,'fdt_matrix2sub.dot')
# if we choose to do a fsl-like clustering: one need to use percent option
if option_cluster == 4:
    option_percent = 1
#
#loading the datas : seed masks, target masks and connectivity matrix fdt_matrix2.dot
#loading the seed
seedH = nib.load(path_seed)
#
seed = seedH.get_data()
#
# m1 : how many non-zeros in seed (could have used numpy nonzero function...)
m1 = sum(sum(sum(1 for i in row if i > 0) for row in row2) for row2 in seed)
#
mask = seed.astype(np.bool)
#
shape = mask.shape
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
#==============================================================================
# when using option percent 1, one doesn't need to load the connectivity matrix fdt_matrix
# neither the target .nii.gz, one juste need seed_to_*
print 'load connectivity matrix'
if option_percent!=1:
# FULL CONNECTIVITY MATRIX
# connectivity matrix : number of seed voxel X number of target voxels (values=number of pathways)
    connectivity_matrix_file = open(path_connectivity_matrix,"r")
#
    connectivity_matrix = connectivity_matrix_file.read()
#   
    if option_irmf==0:
#
        text = connectivity_matrix.split('\n')
#
        A = np.zeros((len(text),3))
#
        print 'compute connectivity matrix...'
#        
    # Convert the format : X Y #number_of_tracts (matrice A)  -->  matrix #voxel_seed times #voxel_targets (matrice connect)
        for i in range(len(text)-1): 
#
            data = np.fromstring(text[i], dtype=int, sep=" ")        
#
            A[i,0]=data[0]-1
#
            A[i,1]=data[1]-1
#
            A[i,2]=data[2]        
#
            if option_length_tract==1:            
#
                # pre-processing : take into account the length of the tract   
#
                # if the distance A[i,0]->A[i,1] is high: then A[i,2] is renforced         
#
                distance = np.linalg.norm(connect_use2[A[i,0]]-connect_use2[A[i,1]])
#
                # one can then either remove the connection that are too small
                # or weight the diffusion conectivity matrix with the calculated distance            
#
                if option_weight==1:
#
                    if distance < distance_treshold:
#
                        A[i,2]=0
#
                elif option_weight==2:
#
                    # if distance is below the treshold -> the weight of the connection is lowered
#
                    # if distance is above the treshold -> the weight of the connection is increased
#
                    A[i,2] = A[i,2] + alpha*(distance - distance_treshold)
#
                
        #defining sparse matrix from A : connect is the connectivity matrix #seed X #targets
#
        # m2 : how many non-zeros in target 
#       <-> nombre de colonne de la matrice de connectivite (1100 defini lors du subsampling)
        m2=1100
#
        m1=np.nonzero(seed)[0].size
#       matrice de connectivite de la forme seed voxels x target voxels
        connect = sparse.csc_matrix((A[:,2],(A[:,0],A[:,1])),shape=(m1,m2+3),dtype=np.float32)
#
        connect = connect.todense()
#      chargement du systeme de coordonnees
        if use_custom_coordinates==1: 
#           chaque voxel de ce nifti est la coordonnee en X dans le systeme de coordonnee
            niftiX = nib.load(op.join(subject_path,'freesurfer_seg','niftiX.nii.gz')).get_data()
#            chaque voxel de ce nidti est la coordonnee en Y dasn ce systeme de coordonne
            niftiY = nib.load(op.join(subject_path,'freesurfer_seg','niftiY.nii.gz')).get_data()
#               idem Z
            niftiZ = nib.load(op.join(subject_path,'freesurfer_seg','niftiZ.nii.gz')).get_data()
#             
#       regularisation spatiale : ajouter une colonne X,Y,Z a la matrice de connectivite

        print 'regularization : add custom coordinates system'
        for i in range(connect.shape[0]):
#           normaliser entre 0 et 1 la matrice de connectivite
            if float(connect[i,:].max())!=0:
#
                connect[i,:] = connect[i,:]/float(connect[i,:].max())
#           si on choisit d'utiliser le systeme de coordonnee euclidien classique :
            if use_custom_coordinates==0:
#                
                d=connect_use2[i]
#               # on ajoute X,Y,Z
                connect[i,m2] = alpha2*d[0]/shape[0] 
#
                connect[i,m2+1] = alpha2*d[1]/shape[1]
#
                connect[i,m2+2] = alpha2*d[2]/shape[2]
#
            else:
                # si on choisit d'utiliser la parametrisation du corps calleux :
                
                d=connect_use2[i]
#               
                connect[i,m2] = alpha2*niftiX[d[0],d[1],d[2]]/shape[0] 
#
                connect[i,m2+1] = alpha2*niftiY[d[0],d[1],d[2]]/shape[1]
#
                connect[i,m2+2] = alpha2*niftiZ[d[0],d[1],d[2]]/shape[2]
                
#
#   DANS LE CAS IRMF (ne pas utiliser pas stable pour le moment)
    else:
#
        m1=np.nonzero(seed)[0].size
#
        text = connectivity_matrix.split('\n')
#
        m2 = np.fromstring(text[0], dtype=float, sep=" ").size
#
        connect = np.zeros((m1,m2))
#
        for i in range(len(text)-1):
#
            data = np.fromstring(text[i], dtype=float, sep=" ")
#
            connect[i,:]= data

# parcellaisation basee sur un atlas (desikan)
elif option_percent == 1:   
#    
    print 'compute freesurfer-seg based connectivity matrix...'
#    
    cmd_s = 'ls -A1 %s/seeds_to* | wc -l' %(path_tracto)
#
    nb_targets = int(commands.getoutput(cmd_s))-1
#
    cmd_s2 = 'ls -A1 %s/seeds_to* ' %(path_tracto)
#
    names = commands.getoutput(cmd_s2).split('\n')
#
    t = [0 for i in range(nb_targets)]
#

    # loading the target FS masks in a table : each index of
    # table t is a mask, eventually t is a 4D matrix
    for it in range(nb_targets):
#
        t[it] = nib.load(names[it]).get_data()
# 
    sh = seed.shape
#  
    Mat_proportion = np.zeros((m1, nb_targets))
#
    index_voxel = 0
## a changer enutilisant np.where avec connectu_use2
    # for each voxel of the seed, let us look the connectivity which each seed_to_ mask
    for k in range(sh[2]):
#
        for j in range(sh[1]):
#
            for i in range(sh[0]-1, -1, -1):
#
                if seed[i][j][k] > 0:
#
                    sm=0
#
                    for l in range(nb_targets):   
#
                        sm = sm + t[l][i][j][k]
#
                        Mat_proportion[index_voxel, l] = t[l][i][j][k]
#
                    Mat_proportion[index_voxel, :] = Mat_proportion[index_voxel, :]/sm
#
                    index_voxel = index_voxel+1
#
                
    print 'cross-correlation...'
#
    corr_connectivity = 1+np.corrcoef(Mat_proportion)  
#
    connect = Mat_proportion

# CLUSTERING STAGE
# USE WARD
if option_cluster == 1:
 
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

# OBSOLETE : DON'T USE
elif option_cluster == 2:
# perform the k-means clustering : the labels for each voxel seed are in table : labelsf
    print 'kmeans...'
#
    k_means = KMeans(init = 'k-means++', n_clusters = nb_cluster, n_init = 10)
#
    k_means.fit(connect)
#
    labelsf = k_means.labels_
# USE IT INSTEAD
elif option_cluster == 3: 
#
    print 'cross-correlation, for building similarity matrix.'
#   cross correlation de la matrice de connectivite
    corr_connectivity = 1+np.corrcoef(connect)  
#
    corr_connectivity = np.nan_to_num(corr_connectivity)
#
    print 'spectral clustering...'
#
    labelsf = spectral_clustering(corr_connectivity, n_clusters = nb_cluster)
    
elif option_cluster == 4:
# FSL-like clustering : choose the target mask with the maximum probability of connection
    print 'hard clustering...'
#
    labelsf = np.zeros((m1))    
#
    for i in range(m1):
#
        max_parcelle = -1
#
        p = 1
#
# pick the freesurfer mask with highest connection        
        for hj in Mat_proportion[i, :]:
#
            if hj > max_parcelle:
#
                max_parcelle = hj
#
                indx = p
#
            p = p+1
#
        labelsf[i] = indx
#
if option_print_matrix==1:
    # One can visualize the mean distribution of the mean connectivity pattern for
    # each parcel (cluster)
    target2H=nib.load(path_target)
#
    target2=target2H.get_data()       
# 
    # reading the coordinate file : linking the subsampling coordinates with originak coordinates
    coords=np.zeros((target2.shape))          
#
    coord_matrix_file = open(coord_matrix,"r")
#
    coord_matrix_ = coord_matrix_file.read()
#
    text = coord_matrix_.split('\n')
#
    for i in range(len(text)-1): 
#
        data = np.fromstring(text[i], dtype=int, sep=" ")  
#
        coords[data[0],data[1],data[2]]=data[3]
#
       
    for parcel in range(nb_cluster):
#
        # compute the mean connectivity pattern
        inds=np.where(labelsf==parcel)
#
        D = np.zeros(connect.shape[1])    
#
        for i in range(connect.shape[1]):     
#
            for j in inds[0]:
#
                D[i] = D[i] + connect[j, i]
#
            D[i] = D[i]/inds[0].size  
#
        # compute the nifti iage corresponding to this pattern    
        sh = target2.shape
#
        parcellation_data = np.zeros((sh[0], sh[1], sh[2]),dtype='float32')
#
        index_voxel = 0
## a changer enutilisant np.where avec connectu_use2
        for k in range(sh[2]):
#
            for j in range(sh[1]):
#
                for i in range(sh[0]-1, -1, -1):
#
                    if target2[i, j, k] > 0:
                        
                        parcellation_data[i, j, k] = D[coords[i,j,k]]
#
                    
        #thresholding 
        parcellation_data_scaled = parcellation_data/np.float(parcellation_data.max())
#
        parcellation_data_scaled[np.where(parcellation_data_scaled<treshold)]=0
        #saving
        sFinal=nib.Nifti1Image(parcellation_data_scaled, target2H.get_affine())
#
        nib.save(sFinal, path_tracto+'/parcel'+str(option_percent)+str(option_cluster)+str(parcel)+'.nii.gz')
        #plot(D)
    
    coord_matrix_file.close()
    # print covariance matrix
    #for parcel in range(nb_clusters):
    #    inds=np.where(labelsf==parcel)
    #    cov_matrix = np.zeros((inds[0].size,inds[0].size))
    #    D = np.zeros(surf_connect_mat.shape[1])  
    #    a=0
    #    for i in inds[0]:  
    #        b=0
    #        for j in inds[0]:
    #            for t in range(surf_connect_mat.shape[1]):     
    #                cov_matrix[a,b] = cov_matrix[a,b] + (surf_connect_mat[i, t]-surf_connect_mat[j, t])*(surf_connect_mat[i, t]-surf_connect_mat[j, t])
    #            b=b+1
    #        a=a+1         
    #    imshow(cov_matrix)
        
    #clusterised_mat = np.zeros((m1,m1))
    #to_print = np.zeros((m1,m1))
    #it = 0
    #for c in range(nb_cluster):
    #    for k in labelsf:
    #        if k == c:
    #            clusterised_mat[it, :] = corr_connectivity[k, :]         
    #            to_print[it, :] = Mat_proportion[k, :] 
    #            it=it+1
       
    #it = 0
    #clusterised_mat2 = np.zeros((m1,m1))
    #for c in range(nb_cluster):
    #    for k in labelsf:
    #        if k == c:
    #            clusterised_mat2[:, it] =  clusterised_mat[:, k] 
    #            it=it+1
                    
    
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
 
