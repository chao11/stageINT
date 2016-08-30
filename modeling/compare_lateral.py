import compare_statistic as cs
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import os.path as op
import joblib
from scipy import stats
import numpy as np


def compare_lateral(path, hemisphere, weight, y_file, atlas, norma, compare_options):
    print hemisphere, weight, y_file, atlas, norma

    a_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                            atlas=atlas, lateral=compare_options[0], y=y_file, w=weight)
    a_data = joblib.load(op.join(path, a_file))

    b_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                            atlas=atlas, lateral=compare_options[1], y=y_file, w=weight)
    b_data = joblib.load(op.join(path, b_file))

    if norma in a_data.columns:
        a = a_data[norma].astype(float)
    else:
        print '%s not exists in %s' %(norma, a_file)
        a = pd.Series(np.zeros(a_data.shape[0]), index=a_data.index)

    if norma in b_data.columns:
        b = b_data[norma].astype(float)
    else:
        print '%s not exists in %s' %(norma, b_file)
        b = pd.Series(np.zeros(b_data.shape[0]), index=b_data.index)


    if 'AHS22' in a.index:
        a = a.drop('AHS22')
        b = b.drop('AHS22')
        print 'remove AHS22'

    t, p = stats.ttest_rel(a, b)
    str = cs.get_difference_ttest(compare_options[0], compare_options[1], t, float(p))

    a_score = cs.convert_to_dataframe(a, hemisphere, compare_options[0], atlas, norma, y_file, weight)
    b_score = cs.convert_to_dataframe(b, hemisphere, compare_options[1], atlas, norma, y_file, weight)
    all_score = pd.concat([a_score, b_score])

    return all_score, {'p': p, 't': t, 'result': str}


# main
model_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux_with_AHS22'

weight = 'none'
hemisphere = 'lh'
norma = 'zscore'
atlas ='destrieux'
y = 'rcon_0002'
compare = ['ipsi', 'bilateral']

for y in ['rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']:

    lh_data, t_test_lh= compare_lateral(model_score_path, 'lh', weight, y, atlas, norma, compare)
    rh_data, t_test_rh= compare_lateral(model_score_path, 'rh', weight,  y, atlas, norma, compare)

    All_data = pd.concat([lh_data, rh_data])

    mean = All_data.groupby(['hemi', 'lateral'])['r2'].mean()
    print mean

    """
    ax = sns.boxplot(x="hemi", y="r2", hue="lateral", data=All_data, palette="PRGn")
    ax.set_title('compare %s vs %s for each hemisphere\natlas:%s, norma:%s, weight:%s, lateral:%s'
                 %(compare[0], compare[1], atlas, norma, weight, y))

    ax.set_ylim([-1, 0.5])
    ax.text(-0.25, 0.4, 'p= %.3E\n%s' %(t_test_lh['p'], t_test_lh['result']))
    ax.text(0.75, 0.4, 'p= %.3E\n%s' %(t_test_rh['p'], t_test_rh['result']))
    ax.text(0.5, -1, 'man_R2score:\n%s'%mean)
    """

    ax = All_data.boxplot(by=['hemi','lateral'], column="r2", sym='')
    means = {id+1: round(x, 3) for id, x in enumerate(mean)}
    plt.scatter(means.keys(), means.values())
    for k, v in means.items():
        plt.text(k+0.1, v, v)

    #plt.text(1, 0.3, 'p= %.3f\n%s' %(t_test_lh['p'], t_test_lh['result']))
    #plt.text(3, 0.3, 'p= %.3f\n%s' %(t_test_rh['p'], t_test_rh['result']))
    plt.title('left:p= %.3f, %s;    right:p= %.3f, %s' %(t_test_lh['p'], t_test_lh['result'],t_test_rh['p'], t_test_rh['result'] ))
    plt.ylabel('R2 score')
    plt.savefig('/hpc/crise/hao.c/model_result/compare_parameters/compare_lateral/compare_%s vs %s(%s_%s_%s).png'%(compare[0], compare[1], y, norma, weight))
    plt.show()

