# hao.c
# 12/07/2016


"""

select subject and plot the R2score

"""

import joblib
import os
import os.path as op
import numpy as np
import pandas as pd
import xlrd
import nibabel as nib
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pylab as plt


def get_difference_ttest(a, b, t_statistic, p_value):
    if p_value > 0.05:
        print " t-test: t=%s, p=%E,  %s = %s " % (t_statistic, p_value, a, b)
    else:
        if t_statistic > 0:
            print " t-test: t=%s, p=%E,  %s > %s " % (t_statistic, p_value, a, b)
        else:
            print " t-test: t=%s, p=%E,  %s < %s " % (t_statistic, p_value, a, b)


def plot_predict_real(subject, norma, contrast, predict_file, i_subplot):
    def extract_functional(coord, img_fmri):
        y = []
        for i in coord:
            y.append(img_fmri[i[0], i[1], i[2]])
        return y

    print predict_file
    predict = joblib.load(predict_file)
    file_name = op.splitext(op.basename(predict_file))[0]
    print file_name
    contrast_img = '/hpc/banco/voiceloc_full_database/func_voiceloc/%s/nomask_singletrialbetas_spm12_stats/' \
                   'resampled_fs5.3_space/%s.nii' % (subject, contrast)
    print contrast_img
    img = nib.load(contrast_img).get_data()

    y = np.asarray(extract_functional(predict[0], img))
    score = r2_score(y, predict[1])
    print score

    plt.subplot(2, 3, i_subplot)
    plt.plot(predict[1], y, '.')
    plt.xlabel('predict')
    plt.ylabel('actual respons')
    plt.title( '%s R2 score: %f '% (norma, score))


def getFileName(path):
    f_list = os.listdir(path)
    jl_list = []
    for i in f_list:
        if op.splitext(i)[1] == '.jl':
            print i
            jl_list.append(i)
    return jl_list


# todo: add altas, model to parameters
def set_compare_parameter(dict_para, fix_para_list, compare_para, output_base_dir, kind=''):
    print "\n\ncompare the result of %s model" % compare_para

    score = {}
    abscent_list = []
    for h in dict_para[fix_para_list[0]]:
        for j in dict_para[fix_para_list[1]]:
            for k in dict_para[fix_para_list[2]]:
                for l in dict_para[fix_para_list[3]]:

                    for m in dict_para[compare_para]:
                        p = {fix_para_list[0]:h, fix_para_list[1]:j, fix_para_list[2]: k, fix_para_list[3]:l,
                             compare_para: m}
                        # print p
                        p[compare_para] = m

                        jl_file_name = 'r2score_{}_{}_{}_{}_{}_weighted{}.jl'\
                            .format(p['hemisphere'], parcel_altas, p['lateral'], model, p['y_file'], p['weight'])
                        # print jl_file_name

#                       data jl containt the normalisation result of each model:
                        data = joblib.load(op.join(output_base_dir, jl_file_name))
                        norma = p['normalise']
#
                        if norma in data.columns:
                            score[m] = data[norma]
#
                        else:
                            # print " the R2 score for {} is not calculated yet,\nset the score at 0. " .format(p)
                            score[m] = pd.Series(np.zeros(data.shape[0]), index=data.index)
                            abscent_list.append(p)

                    compare = pd.DataFrame(score)

                    # remove AHS22 for right hemisphere:
                    if compare_para == 'hemisphere' or p['hemisphere'] == 'rh':
                        compare = compare[1:]
#
                    mean_score_of_70_subj = compare.mean(axis=0)
#                   check the type of the data, data type must be numerid for plotting
                    data_type = compare.dtypes
                    convert_col = data_type[data_type == object].index
                    if len(convert_col) > 0:
                        compare[convert_col] = compare[convert_col].convert_objects(convert_numeric=True)

#                   get title
                    string = ''
                    for key, value in p.items():
                        if key != compare_para:
                            string = string + key + '_' + value + '_'
                    img_title = "compare_%s_model(%s)" % (compare_para, string[:-1])
                    # print img_title

                    plt.figure(figsize=(20, 10))
#                   plot image on subject
                    if kind == '':
                        compare.plot(figsize=(20, 10))
                        plt.xlabel('subject')
                        plt.ylabel('R2 score')
                        plt.legend(loc='lower left')
                        plt.text(50, round(np.amin(compare.min(axis=0))), 'mean of R2 score: \n {}'.format(mean_score_of_70_subj))

                    elif kind == 'box':
                        #sns.set(style="ticks")
                        #sns.boxplot(data=compare)
                        bp_dict = compare.boxplot()

                        # get dictionary returned from boxplot
                        for line in bp_dict['medians']:
                            x, y = line.get_xydata()[1]
                            plt.text(x,y, '%.3f'%y, verticalalignment='top')  # on the right of the box

                        # get the annotate subject from 'fliers:
                        for flier in bp_dict['fliers']:
                            x = flier.get_xdata()   # x label = datafram.col
                            y = flier.get_ydata() # score
                            y.sort()
                            if len(x)>0:
                                for i in range(len(x))[0:2]:    # just point 2 annotate subjects

                                    col = compare.columns[x[i]-1]
                                    index = np.where(compare[col] == y[i])
                                    subject = compare.index[index[0]].values
                                    # print "annotate subject: %s" % subject
                                    # plt.annotate(subject, xy=(x[i], y[i]), xytext=(x[i]+0.08, y[i]))

                                    if i>0:
                                        plt.annotate(subject, xy=(x[i], y[i]), xytext=(x[i]+0.1, y[i]+0.1),
                                                     arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
                                    else:
                                        plt.annotate(subject, xy=(x[i], y[i]), xytext=(x[i]+0.03, y[i]),
                                                     arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))

                    plt.title(img_title)
                    # plt.show()

#                   save image
                    output = op.join(output_base_dir, 'compare_%s' % compare_para)
                    if not op.isdir(output):
                        os.mkdir(output)
                    save_img = op.join(output, "%s_%s.png" % (kind, string[:-1]))
                    # plt.savefig(save_img)
                    # print save_img

                    for set1 in range(len(compare.columns)):
                        for set2 in range(len(compare.columns)):
                            if set != set2:
                                t, proba = stats.ttest_rel(compare.ix[:, set1], compare.ix[:, set2])
                                print h, j, k, l
                                print compare.mean(axis=0)
                                get_difference_ttest(compare.columns[set1], compare.columns[set2], t, proba)





#TODO: add compare the mean score and variance, error bar plot
    return abscent_list


# ============================= main ===================================================================================
parcel_altas = 'destrieux'
model = 'connmat'
"""
dict_p = {'hemisphere': ['lh', 'rh'],
                   'normalise': ['none', 'waytotal', 'norm', 'sum', 'partarget', 'zscore'],
                   'lateral': ['ipsi', 'bilateral'],
                   'weight': ['none', 'distance'],
                   'y_file': ['rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']}
"""
dict_p = {'hemisphere': ['lh', 'rh'],
                   'normalise': [ 'zscore', 'norm', 'sum', 'partarget'],
                   'lateral': ['ipsi', 'bilateral'],
                   'weight': ['none', 'distance'],
                   'y_file': ['rspmT_0002', 'rcon_0002']}




# compare one parameter for all subjects:
#parameter_list = dict_p.keys()
parameter_list = ['normalise', 'lateral', 'weight', 'y_file']
compare_parameter = 'lateral'
list = []

for compare_parameter in parameter_list:
    fix_para = {k: v for k, v in dict_p.items() if k != compare_parameter}
    fix_para_list = fix_para.keys()
    output_dir = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux/'

    abscent_list = set_compare_parameter(dict_p, fix_para_list, compare_parameter, output_dir, 'box')
    print '\n'
    list.append(abscent_list)

print "\n\nabscent score list: %s\n"%len(list), list



hemi = 'lh'
lateeral = 'ipsi'

data_lh_rcon_ipsi = joblib.load('/hpc/crise/hao.c/model_result/tracto_volume/r2score_lh_destrieux_ipsi_connmat_rcon_0002_weightednone.jl')
data_rh_rcon_ipsi = joblib.load('/hpc/crise/hao.c/model_result/tracto_volume/r2score_rh_destrieux_ipsi_connmat_rcon_0002_weightednone.jl')
stats.ttest_rel(data_rh_rcon_ipsi['norm'], data_rh_rcon_ipsi['zscore'])


data_lh_rspm_ipsi = joblib.load('/hpc/crise/hao.c/model_result/tracto_volume_destrieux/r2score_lh_destrieux_ipsi_connmat_rspmT_0002_weightednone.jl')


path = '/hpc/crise/hao.c/model_result/tracto_volume_desikan'
file_list  = getFileName(path)
for file in file_list:
    score = joblib.load(op.join(path, file))
    print file
    print score.min(axis=0)
    print score.ix[['AHS22', 'JHN01']]
    print '\n'