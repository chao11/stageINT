"""
WRONG script

this script sompares the paramaters using paired t_test of R2 score,
however the score is the mean score of subjects, thus the t-statistic comparison is acrosse model, not a fixed set of parameters

"""

from scipy import stats
import numpy as np
import joblib
import os.path as op
import os
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


def getFileName(path):
    f_list = os.listdir(path)
    jl_list = []
    for i in f_list:
        if op.splitext(i)[1] == '.jl':
            print i
            jl_list.append(i)

    print len(jl_list)
    return jl_list


def get_difference_ttest(a, b, t_statistic, p_value):
    if p_value > 0.05:
        print " t=%.2f, p=%.3f,  %s = %s " % (t_statistic, p_value, a, b)
        str = '%s = %s '% (a, b)
        return str
    else:
        if t_statistic > 0:
            print " t=%.2f, p=%.3f,  %s > %s " % (t_statistic, p_value, a, b)
            str = '%s > %s '% (a, b)
            return str

        else:
            print " t=%.2f, p=%.3f,  %s < %s " % (t_statistic, p_value, a, b)
            str = '%s < %s '% (a, b)
            return str



# convert the series of R2 score into dataframe completed with the other parameters
def convert_to_dataframe_for_contrast(series_score, hemisphere, lateral, atlas, norma, contrast_type, contrast_number, weight):
    series_score = pd.DataFrame(series_score)
    # rename the column as r2
    series_score = series_score.rename(columns={series_score.columns[0]: 'r2'})
    series_score['hemi'] = pd.Series(hemisphere, index=series_score.index)
    series_score['weight'] = pd.Series(weight, index=series_score.index)
    series_score['norma'] = pd.Series(norma, index=series_score.index)
    series_score['lateral'] = pd.Series(lateral, index=series_score.index)
    series_score['contrsat_number'] = pd.Series(contrast_number, index=series_score.index)
    series_score['contrast_type'] = pd.Series(contrast_type, index=series_score.index)
    series_score['targets'] = pd.Series(atlas, index=series_score.index)

    return series_score

def convert_to_dataframe(series_score, hemisphere, lateral, atlas, norma,y_file, weight):
    series_score = pd.DataFrame(series_score)
    # rename the column as r2
    series_score = series_score.rename(columns={series_score.columns[0]: 'r2'})
    series_score['hemi'] = pd.Series(hemisphere, index=series_score.index)
    series_score['weight'] = pd.Series(weight, index=series_score.index)
    series_score['norma'] = pd.Series(norma, index=series_score.index)
    series_score['lateral'] = pd.Series(lateral, index=series_score.index)
    series_score['y_file'] = pd.Series(y_file, index=series_score.index)
    series_score['targets'] = pd.Series(atlas, index=series_score.index)

    return series_score


# compare normalisation
def compare_normalised(model_score_path, hemisphere, weight, lateral, atlas, y_file, list_compare):

    jl_file_name = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere, atlas=atlas,
                                                                                     lateral=lateral,y=y_file,w=weight)
    print jl_file_name
    jl_path = op.join(model_score_path, jl_file_name)
    if op.isfile(jl_path):
        jl = joblib.load(jl_path)

        print "%s vs %s " %(list_compare[0], list_compare[1])
        a = jl[list_compare[0]]
        b = jl[list_compare[1]]
        a = a.astype(float)
        b = b.astype(float)

        if 'AHS22' in a.index:
            a = a.drop('AHS22')
            b = b.drop('AHS22')

        t, p = stats.ttest_rel(a, b)
        str = get_difference_ttest(list_compare[0],list_compare[1], t, float(p))

        a_score = convert_to_dataframe(a, hemisphere, lateral, atlas, list_compare[0], y_file, weight)
        b_score = convert_to_dataframe(b, hemisphere, lateral, atlas, list_compare[1], y_file, weight)
        all_score = pd.concat([a_score, b_score])

        return all_score, {'p':p, 't':t, 'result':str}
    else:
        print '%s not exists'%jl_path



# compare contrast:
def contrast_VS_t(path, hemisphere, weight, lateral, atlas, norma):
    print hemisphere, weight, lateral, atlas, norma

    for i in range(1, 5):
        rconfile = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                                atlas=atlas,lateral=lateral,y='rcon_000%d'%i,w=weight)
        rcon = joblib.load(op.join(path, rconfile))

        t_value = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                                atlas=atlas,lateral=lateral,y='rspmT_000%d'%i,w=weight)
        rspmT = joblib.load(op.join(path, t_value))

        a = rcon[norma].astype(float)
        b = rspmT[norma].astype(float)

        t, p = stats.ttest_rel(a, b)
        get_difference_ttest('rcon%d'%i,'rspmT%d'%i, t, float(p))

    return a, b, t, p


def distance(path, hemisphere, lateral, atlas, norma, y_file):
    print hemisphere, lateral, atlas, norma, y_file

    distance_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                            atlas=atlas,lateral=lateral,y=y_file,w='distance')
    distance = joblib.load(op.join(path, distance_file))

    no_distance_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                            atlas=atlas,lateral=lateral,y=y_file,w='none')
    no_distance = joblib.load(op.join(path, no_distance_file))

    a = distance[norma].astype(float)
    b = no_distance[norma].astype(float)

    t, p = stats.ttest_rel(a, b)
    str = get_difference_ttest('distance','no distance', t, float(p))

    a_dataframe = convert_to_dataframe(a, hemisphere, lateral, atlas, norma, y_file, 'distance')
    b_dataframe = convert_to_dataframe(b, hemisphere, lateral, atlas, norma, y_file, 'no_distance')
    all_data = pd.concat([a_dataframe, b_dataframe])


    return all_data, {'p':p, 't':t, 'result':str}


def compare_atlas(hemisphere, weight, lateral, y_file, norma):
    print hemisphere, weight, lateral, y_file, norma
    destrieux_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux'
    destrieux_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,

                                                                           atlas='destrieux',lateral=lateral,y=y_file,w='distance')

    aparcaseg_path = '/hpc/crise/hao.c/model_result/tracto_volume_aparcaseg_with_AHS22'
    aparcaseg_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                           atlas='aparcaseg',lateral=lateral,y=y_file,w='distance')
    if op.isfile(op.join(destrieux_path, destrieux_file)):
        destrieux = joblib.load(op.join(destrieux_path, destrieux_file))
    if op.isfile(op.join(aparcaseg_path, aparcaseg_file)):
        aparcaseg = joblib.load(op.join(aparcaseg_path, aparcaseg_file))

    if norma in destrieux.columns:
        a = destrieux[norma].astype(float)
#
    else:
        # print " the R2 score for {} is not calculated yet,\nset the score at 0. " .format(p)
        a = pd.Series(np.zeros(destrieux.shape[0]), index=destrieux.index)
        print "not exists", destrieux_file, norma

    if norma in aparcaseg.columns:
        b = aparcaseg[norma].astype(float)
#
    else:
        # print " the R2 score for {} is not calculated yet,\nset the score at 0. " .format(p)
        b = pd.Series(np.zeros(aparcaseg.shape[0]), index=aparcaseg.index)
        print "not exists", aparcaseg_file, norma

    t, p = stats.ttest_rel(a, b)
    str = get_difference_ttest('destrieux','aparcaseg', t, float(p))

    a_dataframe = convert_to_dataframe(a, hemisphere, lateral, 'destrieux', norma, y_file, weight)
    b_dataframe = convert_to_dataframe(b, hemisphere, lateral, 'aparcaseg', norma, y_file, weight)
    all_data = pd.concat([a_dataframe, b_dataframe])

    return all_data, {'p':p, 't':t, 'result':str}



# jl_list = getFileName(model_score_path)

"""
# ====================================== compare normalisation: ==============================================
weight = 'none'
hemisphere = 'rh'
y_file = 'rcon_0002'
atlas = 'destrieux'

compare_normalised(model_score_path, hemisphere, weight, 'ipis', atlas, y_file)

# ====================================== compare contrast ==================================================
weight = 'none'
hemisphere = 'lh'
lateral = 'ipsi'
norma = 'none'
atlas ='destrieux'

for hemisphere in ['lh', 'rh']:
    for lateral in ['ipsi', 'bilateral']:
        for norma in ['none', 'zscore', 'norm']:

a, b, t, p = contrast_VS_t(model_score_path, hemisphere, weight, lateral, atlas, norma)
compare = pd.DataFrame(a)
compare.join(b, how='outer')
compare
"""


# ================================= compare distance ========================================================
"""
atlas = 'destrieux'
y_file = 'rcon_0001'
norma = 'none'
lateral = 'ipsi'
lh_data, t_test_lh = distance(model_score_path, 'lh', lateral, atlas, norma, y_file)
rh_data, t_test_rh = distance(model_score_path, 'rh', lateral, atlas, norma, y_file)
All_data = pd.concat([lh_data, rh_data])

mean = All_data.groupby(['hemi', 'weight'])['r2'].mean()
print mean

ax = sns.boxplot(x="hemi", y="r2", hue="weight", data=All_data, palette="PRGn")
ax.set_title('compare distance/no_distance model for each hemisphere\natlas:%s, y:%s, nomra:%s, lateral:%s' %(atlas, y_file, norma, lateral))

ax.set_ylim([-1, 0.5])
ax.text(-0.25, 0.4, 'p= %.3E\n%s' %(t_test_lh['p'], t_test_lh['result']))
ax.text(0.75, 0.4, 'p= %.3E\n%s' %(t_test_rh['p'], t_test_rh['result']))
ax.text(0.5, -1, mean)

#plt.savefig('/hpc/crise/hao.c/model_result/compare_distance_%s.png'%y_file)
#plt.show()
"""

"""
y_file = 'rcon_0002'
weight = 'none'
hemisphere = 'rh'
norma = 'zscore'
lateral = 'ipsi'

compare_atlas(hemisphere, weight, lateral,y_file, norma)
lh_data, t_test_lh = compare_atlas('lh', weight, lateral,y_file, norma)
rh_data, t_test_rh = compare_atlas('rh', weight, lateral,y_file, norma)
All_data = pd.concat([lh_data, rh_data])

mean = All_data.groupby(['hemi', 'atlas'])['r2'].mean()
print mean

ax = sns.boxplot(x="hemi", y="r2", hue="weight", data=All_data, palette="PRGn")
ax.set_title('compare targets model for each hemisphere\nweight:%s, y:%s, nomra:%s, lateral:%s' %(weight, y_file, norma, lateral))

ax.set_ylim([-1, 0.5])
ax.text(-0.25, 0.4, 'p= %.3E\n%s' %(t_test_lh['p'], t_test_lh['result']))
ax.text(0.75, 0.4, 'p= %.3E\n%s' %(t_test_rh['p'], t_test_rh['result']))
ax.text(0.5, -1, mean)

plt.show()
"""
"""
for hemisphere in ['lh', 'rh']:
    for lateral in ['ipsi']:
        for norma in [ 'zscore']:
            for weight in ['none']:
                for y_file in ['rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']:
                    compare_atlas(hemisphere, weight, lateral,y_file, norma)
"""





""""
LOSO_mean_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux/LOSO_mean_r2_all_model.jl'
data = joblib.load(LOSO_mean_score_path)
data = data.fillna(0)

print data.columns

print "compare ipsi bilateral:"
ipsi = data[data['lateral'] == 'ipsi']
bilateral = data[data['lateral'] == 'bilateral']
t, p = stats.ttest_rel(ipsi['mean_r2'], bilateral['mean_r2'])
get_difference_ttest('ipsi', 'bilateral', t, p)

print "\ncompare with distance or not:"
distance = data[data['weight'] == 'distance'].mean_r2
no_distance = data[data['weight'] == 'none'].mean_r2
t, p = stats.ttest_rel(distance, no_distance)
get_difference_ttest('distance', 'no_distance', t, p)

print "\ncompare contrast and t_value"
for i in range(1, 5):
    a = 'rcon_000%d' % i
    b = 'rspmT_000%d' % i
    contrast = data[data['y_file'] == a].mean_r2
    t_value = data[data['y_file'] == b].mean_r2
    t, p = stats.ttest_rel(contrast, t_value)
    get_difference_ttest(a, b, t, p)

print "compare each contrast"
for i in range(1, 5):
    for j in range(1, 5):

        contrast1 = data[data['y_file'] == 'rcon_000%d' % i].mean_r2
        contrast2= data[data['y_file'] == 'rcon_000%d' % j].mean_r2
        t, p = stats.ttest_rel(contrast1, contrast2)
        get_difference_ttest('rcon_000%d' % i, 'rcon_000%d' % j, t, p)
    print '\n'

print "compare each t_value"
for i in range(1, 5):
    for j in range(1, 5):
        contrast1 = data[data['y_file'] == 'rspmT_000%d' % i].mean_r2
        contrast2 = data[data['y_file'] == 'rspmT_000%d' % j].mean_r2
        t, p = stats.ttest_rel(contrast1, contrast2)
        get_difference_ttest('rcon_000%d' % i, 'rcon_000%d' % j, t, p)
    print '\n'


print("\ncompare normalisation:")
norma_options = ['none', 'norm', 'sum', 'zscore', 'partarget']

for norma1 in norma_options:
    for norma2 in norma_options:
        if norma1 != norma2:
            a_norma = data[data['norma'] == norma1].mean_r2
            b_norma = data[data['norma'] == norma2].mean_r2
            t, p = stats.ttest_rel(a_norma, b_norma)
            get_difference_ttest(norma1, norma2, t, p)


"""
