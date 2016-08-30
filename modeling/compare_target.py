import compare_statistic as cs
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import os.path as op
import joblib
from scipy import stats
import  numpy as np


def compare_atlas(hemisphere, weight, lateral, y_file, norma, compare_options):
    print hemisphere, weight, lateral, y_file, norma
    dict = {}

    destrieux_GM = '/hpc/crise/hao.c/model_result/tracto_volume_%s' %compare_options[0]
    a_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                        atlas=compare_options[0], lateral=lateral, y=y_file, w=weight)

    b_path = '/hpc/crise/hao.c/model_result/tracto_volume_%s'%compare_options[1]
    b_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                        atlas=compare_options[1], lateral=lateral, y=y_file, w=weight)

    c_path = '/hpc/crise/hao.c/model_result/tracto_volume_%s' %compare_options[3]
    c_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                        atlas=compare_options[3], lateral=lateral, y=y_file, w=weight)

    d_path = '/hpc/crise/hao.c/model_result/tracto_volume_%s'%compare_options[4]
    d_file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                        atlas=compare_options[4], lateral=lateral, y=y_file, w=weight)

    if op.isfile(op.join(destrieux_GM, a_file)):
        a_jl = joblib.load(op.join(destrieux_GM, a_file))

        if norma in a_jl.columns:
            a = a_jl[norma].astype(float)
        else:
            # print " the R2 score for {} is not calculated yet,\nset the score at 0. " .format(p)
            a = pd.Series(np.zeros(a_jl.shape[0]), index=a_jl.index)
        print "not exists", a_file, norma
    else:
        print op.join(destrieux_GM, a_file), 'not exits'


    if op.isfile(op.join(b_path, b_file)):
        b_jl = joblib.load(op.join(b_path, b_file))
    else:
        print op.join(b_path, b_file), 'not exits'


    if norma in b_jl.columns:
        b = b_jl[norma].astype(float)
#
    else:
        # print " the R2 score for {} is not calculated yet,\nset the score at 0. " .format(p)
        b = pd.Series(np.zeros(b_jl.shape[0]), index=b_jl.index)

        print "not exists", b_file, norma
    """
    if 'AHS22' in a.index:
        a = a.drop('AHS22')
    if 'AHS22' in b.index:
        b = b.drop('AHS22')
    """

    compare = pd.DataFrame(a)
    compare = compare.rename(columns={compare.columns[0]: compare_options[0]})
    compare = compare.join(b, how='inner')
    compare = compare.rename(columns={compare.columns[1]: compare_options[1]})

    t, p = stats.ttest_rel(compare[compare_options[0]], compare[compare_options[1]])

    str = cs.get_difference_ttest(compare_options[0], compare_options[1], t, float(p))

    a_dataframe = cs.convert_to_dataframe(a, hemisphere, lateral, compare_options[0], norma, y_file, weight)
    b_dataframe = cs.convert_to_dataframe(b, hemisphere, lateral, compare_options[1], norma, y_file, weight)
    all_data = pd.concat([a_dataframe, b_dataframe])

    return all_data, {'p':p, 't':t, 'result':str}


def get_score(hemisphere, weight, lateral, y_file, norma, target):
    print hemisphere, weight, lateral, y_file, norma, target

    if target == 'destrieux_GM':
        tracto_name ='destrieux'
    elif target=='desikan_GM':
        tracto_name = 'aparcaseg'
    else:
        tracto_name = target

    path = '/hpc/crise/hao.c/model_result/tracto_volume_%s'% tracto_name
    file = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                        atlas=tracto_name, lateral=lateral, y=y_file, w=weight)

    if op.isfile(op.join(path, file)):
        jl = joblib.load(op.join(path, file))

        if norma in jl.columns:
            a = jl[norma].astype(float)
        else:
            # print " the R2 score for {} is not calculated yet,\nset the score at 0. " .format(p)
            a = pd.Series(np.zeros(jl.shape[0]), index=jl.index)
        print "not exists", file, norma

        if 'AHS22' in a.index:
            a = a.drop('AHS22')

        a_dataframe = cs.convert_to_dataframe(a, hemisphere, lateral, target, norma, y_file, weight)

        print a_dataframe.shape
        return a_dataframe
    else:
        print op.join(path, file), 'not exits'




# main

weight = 'none'
lateral = 'ipsi'
norma = 'zscore'
y_file ='rcon_0004'

for y_file in ['rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']:




    df = {}
    for i in ['desikan_GM', 'desikan_WM', 'destrieux_GM', 'destrieux_WM']:

        df[i] = get_score('lh', weight, lateral, y_file, norma, i)

    All_data = pd.concat([df[k] for k in df.keys()])

    print All_data.shape, All_data.columns

    ax = All_data.boxplot(column='r2', by='targets')
    mean = All_data.groupby(['targets'])['r2'].mean()
    print mean
    means = {id+1: round(x, 3) for id, x in enumerate(mean)}
    ax.scatter(means.keys(), means.values())
    for k, v in means.items():
        ax.text(k+0.3, v, v)



    t, p = stats.ttest_rel(df['desikan_WM'].r2, df['desikan_GM'].r2 )
    string = cs.get_difference_ttest('desikan_WM', 'desikan_GM', t, p)
    print(string)

    plt.text(1, -0.6, 't=%f, p= %f, \n%s' %(t,p, string))

    compare = pd.DataFrame(df['destrieux_WM'].r2)
    compare = compare.rename(columns={compare.columns[0]: 'destrieux_WM'})
    compare = compare.join(df['destrieux_GM'].r2, how='inner')
    compare = compare.rename(columns={compare.columns[1]: 'destrieux_GM'})

    t, p = stats.ttest_rel(compare['destrieux_WM'], compare['destrieux_GM'])
    string = cs.get_difference_ttest('destrieux_WM', 'destieux_GM', t, p)
    print(string)

    plt.text(3, -0.6, 't=%f, p= %f, \n%s' %(t,p, string))

    compare = pd.DataFrame(df['desikan_GM'].r2)
    compare = compare.rename(columns={compare.columns[0]: 'desikan_GM'})
    compare = compare.join(df['destrieux_GM'].r2, how='inner')
    compare = compare.rename(columns={compare.columns[1]: 'destrieux_GM'})

    t, p = stats.ttest_rel(compare['desikan_GM'], compare['destrieux_GM'])
    string = cs.get_difference_ttest('desikan_GM', 'destieux_GM', t, p)
    print(string)
    plt.text(1, -1.1, 't=%f, p= %f, \n%s' %(t,p, string))


    compare = pd.DataFrame(df['desikan_WM'].r2)
    compare = compare.rename(columns={compare.columns[0]: 'desikan_WM'})
    compare = compare.join(df['destrieux_WM'].r2, how='inner')
    compare = compare.rename(columns={compare.columns[1]: 'destrieux_WM'})

    t, p = stats.ttest_rel(compare['desikan_WM'], compare['destrieux_WM'])
    string = cs.get_difference_ttest('desikan_WM', 'destieux_WM', t, p)
    print(string)
    plt.text(3, -1.1, 't=%f, p= %f, \n%s' %(t,p, string))




    plt.ylabel('R2 score')
    plt.savefig('/hpc/crise/hao.c/model_result/compare_parameters/compare_targets/withoutFlyers_compare_targets_%s_%s.png' %(y_file, norma))

    plt.show()
