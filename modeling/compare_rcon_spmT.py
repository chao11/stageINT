import compare_statistic as cs
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import os.path as op
import joblib
from scipy import stats


# compare contrast:
def contrast_VS_t(path, hemisphere, weight, lateral, atlas, norma, compare_options):
    print hemisphere, weight, lateral, atlas, norma

    rconfile = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                            atlas=atlas,lateral=lateral, y=compare_options[0], w=weight)
    rcon = joblib.load(op.join(path, rconfile))

    t_value = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                            atlas=atlas,lateral=lateral, y=compare_options[1], w=weight)
    rspmT = joblib.load(op.join(path, t_value))

    a = rcon[norma].astype(float)
    b = rspmT[norma].astype(float)

    if 'AHS22' in a.index:
        a = a.drop('AHS22')
        b = b.drop('AHS22')

    t, p = stats.ttest_rel(a, b)
    str = cs.get_difference_ttest('rcon%d'%i,'rspmT%d'%i, t, float(p))

    a_score = cs.convert_to_dataframe(a, hemisphere, lateral, atlas, norma=norma, y_file=compare_options[0], weight= weight)
    b_score = cs.convert_to_dataframe(b, hemisphere, lateral, atlas, norma=norma, y_file=compare_options[1], weight= weight)
    all_score = pd.concat([a_score, b_score])

    return all_score, {'p':p, 't':t, 'result':str}


# main
model_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux_with_AHS22'

weight = 'none'
hemisphere = 'lh'
lateral = 'ipsi'
norma = 'zscore'
atlas ='destrieux'


df = {}
for hemisphere in ['lh', 'rh']:
    data = {}
    for i in range(1, 5):
        rconfile = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                                atlas=atlas,lateral=lateral, y='rcon_000%d'%i, w=weight)
        rcon = joblib.load(op.join(model_score_path, rconfile))

        t_value = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere,
                                                                                atlas=atlas,lateral=lateral, y='rspmT_000%d'%i, w=weight)
        rspmT = joblib.load(op.join(model_score_path, t_value))

        a = rcon[norma].astype(float)
        b = rspmT[norma].astype(float)

        if 'AHS22' in a.index:
            a = a.drop('AHS22')
            b = b.drop('AHS22')

        data ['rcon_000%d'%i] = cs.convert_to_dataframe_for_contrast(a, hemisphere, lateral, atlas, norma, 'rcon', i, weight)
        data ['rspmT_000%d'%i]= cs.convert_to_dataframe_for_contrast(b, hemisphere, lateral, atlas, norma, 'rspmT', i, weight)

    df[hemisphere]= pd.concat([data[k] for k in data.keys()])


All_data = pd.concat([df[k] for k in df.keys()])
All_data['r2'] = All_data['r2'].astype(float)
print All_data.shape, All_data.columns, All_data['r2'].dtypes


mean = All_data.groupby(['hemi', 'contrast_type', 'contrsat_number'])['r2'].mean()
print mean.lh


f = sns.factorplot(x='contrsat_number', y='r2', data=All_data, row='hemi', hue='contrast_type')
axes = f.axes


means = {id+1: round(x, 3) for id, x in enumerate(mean)}
plt.scatter(means.keys(), means.values())
for k, v in means.items():
    plt.text(k+0.1, v, v)

plt.show()