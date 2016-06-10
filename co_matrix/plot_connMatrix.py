import nibabel as nib
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pylab as plt


def original_matrix(subjects_list, hemisphere, altas, target_number):
    # plot the oroginal connectivity matrix
    arr = np.empty((0, target_number), float)

    for subject in subjects_list[0:1]+ subjects_list[2:5]:

        connect_jl = joblib.load('/hpc/crise/hao.c/data/{}/tracto/{}_STS+STG_{}/conn_matrix_seed2parcels.jl'.format(subject, hemisphere.upper(), altas))
        matrix = connect_jl[0]
        arr = np.append(arr, matrix, axis=0)

    plt.imshow(arr,aspect='auto' ,interpolation='nearest')
    plt.colorbar()

    plt.ylabel('voxels (all subjects)')
    plt.xlabel('targets')
    plt.title('%s %s conn_matrix no normalised' %(altas, hemisphere))
    plt.show()
    plt.savefig('/hpc/crise/hao.c/resultat_images/%s_%s_conn_matrix.pdf'%(hemisphere, altas))


root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
#hemisphere = str(sys.argv[1])
hemisphere = 'lh'

Nb_subject = len(subjects_list)
options = ['norm1','norm2','standard','MinMax']

#original_matrix(subjects_list, hemisphere, 'aparcaseg', 79)
original_matrix(subjects_list, 'lh', 'target_mask_aparcaseg_python', 87)


"""
for option in options[3:]:
    plt.figure()

    arr = np.empty((0,163),float)

    for subject in subjects_list:


        matrix = joblib.load('/hpc/crise/hao.c/data/{}/tracto/{}_STS+STG_destrieux/conn_matrix_norma_{}.jl'.format(subject,hemisphere.upper(),option))
        arr = np.append(arr,matrix,axis = 0)


   # plt.imshow(np.log(arr),aspect='auto' ,interpolation='nearest')
    plt.imshow(arr,aspect='auto' ,interpolation='nearest')
    plt.colorbar()
    plt.title('normalize each voxel(row): %s' %option)
    plt.ylabel('voxels (all subjects)')
    plt.xlabel('targets')

    plt.savefig('/hpc/crise/hao.c/resultat_images/conn_matrix_row_norma_{}.png'.format(option))


"""



"""
# for one subject

plt.subplot(221)
plt.imshow(connect,aspect='auto' ,interpolation='nearest')
plt.title('original matrix')
plt.colorbar()

plt.subplot(222)
plt.imshow(matrix,aspect='auto' ,interpolation='nearest')
plt.colorbar()
plt.title('normalized matrix')

plt.subplot(223)
plt.imshow(np.exp(matrix),aspect='auto' ,interpolation='nearest')
plt.colorbar()
plt.title('exp matrix')

plt.subplot(224)
plt.imshow(np.log(matrix),aspect='auto' ,interpolation='nearest')
plt.colorbar()
plt.title('abs of log matrix')

plt.show()
"""