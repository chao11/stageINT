import numpy as np
import nibabel as nib
import os.path as op
from sklearn import cross_validation, linear_model


# define the parameters for modeling
class SetModelParameter:
    hemisphere = ''
    lateral = ''
    cross_validation = ''
    normalize_option = ''
    weight = ''
    y_contrst = ''
    altas = ''
   # subject_list = []
    target_path = ''

    def __init__(self, hemi, lat, cv, norma, w, y, alt, tracto_dir):

        self.hemisphere = hemi
        self.lateral = lat
        self.cross_validation = cv
        self.normalize_option = norma
        self.weight = w
        self.y_contrst = y
        self.altas = alt
        self.tracto_dir = tracto_dir
       # self.x = data[1]
       # self.y = data[2][:, contrast_list.index(y)]
       # self.subject_list = data[0]

# TODO: need target path?
    def lateral_model(self, target_path):

        def ipsi_model(hemisphere, target_path):

            print 'use only the connection probabilities of ' + hemisphere
            target_label = np.unique(nib.load(target_path).get_data())[1:]

            if hemisphere == 'lh':
                label = range(0, 40) + range(1001, 1036) + range(3001, 3036) + range(11101, 11176) + range(13101, 13176)\
                        + [251, 252, 253, 254, 255]
            else:
                label = range(40, 80) + range(2001, 2036) + range(4001, 4036) + range(12101, 12176) + range(14101, 14176)\
                        + [251, 252, 253, 254, 255]

            target = []
            columns = []
            # get the targets for ipsilateral:
            for index, i in enumerate(target_label):
                if i in label:
                    target.append(i)
                    columns.append(index)
            print 'ipsilateral targets ', target

            return columns, target

        if self.lateral == 'ipsi':
            print "use the connectivity of ipsilateral " + self.hemisphere
            # take one of the target mask
            columns, target_label = ipsi_model(self.hemisphere, target_path)
        else:
            target_label = np.unique(nib.load(target_path).get_data())[1:]
            columns = range(0, len(target_label))
            print 'use the connectivity of bilateral'

        return columns, target_label


# normalize the connectivity matrix
class Normalization(SetModelParameter):

    def __init__(self, hemi, lat, cv, norma, w, y, alt, tracto_dir, data, subject):
        SetModelParameter.__init__(self,hemi, lat, cv, norma, w, y, alt, tracto_dir)
        self.x = data
        self.option = SetModelParameter.normalize_option
        self.subject = subject
        root_dir = '/hpc/crise/hao.c/data'
        target_name = '{}_target_mask_{}_165.nii.gz'.format(SetModelParameter.hemisphere, SetModelParameter.altas)
        self.target_path = op.join(root_dir, subject, 'freesurfer_seg', target_name)
        self.waytotal_file = op.join(root_dir, subject, SetModelParameter.tracto_dir, 'waytotal')

    def normaliser(self):
    #   normalize by the norm
        if self.option == 'norm':
            from sklearn.preprocessing import normalize
            x_norma = normalize(self.x, norm='l2')

    #   normalize by the sum of the row, ( normalized matrix sum to 1 )
        elif self.option == 'sum': # normalize sum to 1:
            from sklearn.preprocessing import normalize
            x_norma = normalize(self.x, norm='l1')

    #   normalize each row by z-score : (x-mean)/std
        elif self.option == 'zscore':
            from scipy import stats
            x_norma = stats.zscore(self.x, axis=1)
            # set the nan to 0
            x_norma[np.isnan(x_norma)] = 0

    # normalize each nulber by the number of voxels of target region
        elif self.option == 'partarget':
            # target_path = op.join(root_dir, subject, 'freesurfer_seg', )
            col, target_label = SetModelParameter.lateral_model(self.target_path)
            target_mask = nib.load(self.target_path).get_data()

            # target_label is bilateral or ipsilateral
            # label = np.unique(target)[1:]
            target_size = []
            for i in target_label:
                size = len(target_mask[target_mask == i])
                target_size.append(size)
            # divide each number in the matrix by the size of the target region
            x_norma = np.divide(self.x, target_size)

        elif self.option == 'waytotal':
            with open(self.waytotal_file, 'r')as f:
                file = f.read()
                text = file.split('\n')
                nb_waytotal = np.fromstring(text[0], dtype=int, sep=" ")
                print "waytotal: ", nb_waytotal
            x_norma = self.x/nb_waytotal

        elif self.option == 'none':
           # print ('no normalization')
            x_norma = self.x
        return x_norma


