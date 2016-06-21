#! /usr/bin/python -u
# coding=utf-8


# evaluate the clustering by using the silhouette coefficiant

import sys
import os.path as op
import numpy as np
import nibabel as nib
from sklearn import metrics
import joblib
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
subject = str(sys.argv[1])
hemisphere = str(sys.argv[2])
#nb_cluster = int(sys.argv[3])
norma = str(sys.argv[3])


"""
subject = 'AHS22'
hemisphere = 'lh'


root_dir = '/hpc/crise/hao.c/data'
# define directories
subject_dir = op.join(root_dir,subject)
fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
output_tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemisphere.upper()))


cluster = np.arange(3,9,2)
options = ['none','norm1','norm2','MinMax']

for norma in options:

    avg = []
    for nb_cluster in cluster:
        print('Processing subject {} for {},cluster={}, norma: {}...'.format(subject,hemisphere,nb_cluster,norma))

        ward_outout = op.join(output_tracto_dir,'ward_{}_{}.jl'.format(nb_cluster,norma))
        # load connectivity matrix:
        if norma=='none':
            conn_matrix_path = op.join(output_tracto_dir,'conn_matrix_seed2parcels.jl')
            conn_matrix = joblib.load(conn_matrix_path)
            connect = conn_matrix[0]

        else:
            conn_matrix_path = op.join(output_tracto_dir,'conn_matrix_norma_{}.jl'.format(norma))
            connect = joblib.load(conn_matrix_path)

        # connect = np.exp(connect)
        ward = joblib.load(ward_outout)
        labelsf = ward.labels_

        #=========== evoluation the clustering ===========================================
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        print ("evaluate clustering")
        silhouette_avg = silhouette_score(connect, labelsf)
        avg.append(silhouette_avg)
        print("For n_clusters =", nb_cluster,"The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(connect, labelsf)
        print sample_silhouette_values.shape,sample_silhouette_values


    plt.plot(cluster,avg,label=norma)


plt.xlabel('number of clusters')
plt.ylabel('silhouette_avg')
plt.legend()

plt.title("average silhouette score")
plt.savefig('{}/cluster_silhouette_{}_2.png'.format(output_tracto_dir,norma))


# ========================================================================================================================
# get the size of parcellations : here only compare the none and norm2_cl7
"""
import os
import nibabel as nib
f = open("/hpc/crise/hao.c/RH_sujets_parcellation_info.txt",'a+')

subjects_list = os.listdir(root_dir)
for subject in subjects_list[2:]:
    parcel_none = '/hpc/crise/hao.c/data/{}/tracto/RH_STS+STG_destrieux/{}_RH_seed_parcellisation_cl7none.nii.gz'.format(subject,subject)
    parcel_norm2= '/hpc/crise/hao.c/data/{}/tracto/RH_STS+STG_destrieux/{}_RH_seed_parcellisation_cl7norm2.nii.gz'.format(subject,subject)
    none = nib.load(parcel_none)
    data1 = none.get_data()
    size_none=[]
    for i in np.unique(data1):
        size_none.append(len(data1[data1==i]))
    f.write( "%s parcellation size before normalization: " %(subject))
    f.write('%s\n'%size_none)

    norm2 = nib.load(parcel_norm2)
    data2 = norm2.get_data()
    size_norm2=[]
    for i in np.unique(data2):
        size_norm2.append(len(data2[data2==i]))
    f.write( "%s parcellation size of norm2 cl=7:       " %(subject))
    f.write('%s\n\n' %size_norm2)


f.close()

"""