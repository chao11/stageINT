import nibabel as nib
import matplotlib.pylab as plt
import joblib
import os


root_dir = '/hpc/crise/hao.c/data'
subjects_list = os.listdir(root_dir)
#hemisphere = str(sys.argv[1])
hemisphere = 'lh'

Nb_subject = len(subjects_list)
options = ['norm1','norm2','standard','MinMax']


for option in options:
    plt.figure()
    indx = 1

    for subject in subjects_list:

        matrix = joblib.load('/hpc/crise/hao.c/data/{}/tracto/{}_STS+STG_destrieux/conn_matrix_norma_{}_0.jl'.format(subject,hemisphere.upper(),option))
        #matrix = connect_jl[0]

        plt.subplot(Nb_subject,1,indx)
        plt.imshow(matrix,aspect='auto' ,interpolation='nearest')
        plt.axis('off')

        indx += 1

    # add the title
    plt.subplot(Nb_subject,1,1)
    plt.title('normalize each feature of the conn_matrix: %s' %option)


    plt.savefig('/hpc/crise/hao.c/resultat_images/conn_matrix_feature_norma_{}.png'.format(option))

"""
# plot the oroginal connectivity matrix
indx =1
for subject in subjects_list:

    connect_jl = joblib.load('/hpc/crise/hao.c/data/{}/tracto/{}_STS+STG_destrieux/conn_matrix_seed2parcels.jl'.format(subject,hemisphere.upper()))
    matrix = connect_jl[0]

    plt.subplot(Nb_subject,1,indx)
    plt.imshow(matrix,aspect='auto' ,interpolation='nearest')
    plt.axis('off')

    indx += 1

# add the title
plt.subplot(Nb_subject,1,1)
plt.title('conn_matrix no normalised')


plt.savefig('/hpc/crise/hao.c/resultat_images/conn_matrix.png')
"""