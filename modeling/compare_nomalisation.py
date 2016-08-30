import compare_statistic as cs
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import os.path as op
import joblib
from scipy import stats
import matplotlib


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
        str = cs.get_difference_ttest(list_compare[0],list_compare[1], t, float(p))

        a_score = cs.convert_to_dataframe(a, hemisphere, lateral, atlas, list_compare[0], y_file, weight)
        b_score = cs.convert_to_dataframe(b, hemisphere, lateral, atlas, list_compare[1], y_file, weight)
        all_score = pd.concat([a_score, b_score])

        return all_score, {'p':p, 't':t, 'result':str}
    else:
        print '%s not exists'%jl_path



# ====================================== compare normalisation: ==============================================
weight = 'none'
hemisphere = 'rh'
y_file = 'rcon_0002'
atlas = 'destrieux'
lateral = 'ipsi'

model_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux_with_AHS22'

options = ['none', 'norm', 'sum', 'zscore', 'partarget']

"""
        lh_data, t_test_lh= compare_normalised(model_score_path, 'lh', weight, lateral, atlas, y_file, compare)
        rh_data, t_test_rh= compare_normalised(model_score_path, 'rh', weight, lateral, atlas, y_file, compare)

"""
df = {}
for hemisphere in ['lh', 'rh']:
    jl_file_name = 'r2score_{h}_{atlas}_{lateral}_connmat_{y}_weighted{w}.jl'.format(h=hemisphere, atlas=atlas,
                                                                                         lateral=lateral,y=y_file,w=weight)
    jl_path = op.join(model_score_path, jl_file_name)

    score = joblib.load(jl_path)
    score = score.drop('AHS22')
    data = {}
    for norma in options:
        data[norma]= cs.convert_to_dataframe(score[norma], hemisphere, lateral, atlas, norma, y_file, weight)

    df[hemisphere]= pd.concat([data[k] for k in data.keys()])


All_data = pd.concat([df[k] for k in df.keys()])
All_data['r2'] = All_data['r2'].astype(float)
print All_data.shape, All_data.columns, All_data['r2'].dtypes

mean = All_data.groupby(['hemi', 'norma'])['r2'].mean()
print mean

"""
ax = sns.boxplot(x="hemi", y="r2", hue="norma", data=All_data, palette="PRGn")
ax.set_title('compare normalisation for each hemisphere\natlas:%s, y:%s, weight:%s, lateral:%s'
             %( atlas, y_file, weight, lateral))

ax.set_ylim([-1, 0.5])
ax.text(-0.25, 0.4, 'p= %.3E\n%s' %(t_test_lh['p'], t_test_lh['result']))
ax.text(0.75, 0.4, 'p= %.3E\n%s' %(t_test_rh['p'], t_test_rh['result']))
ax.text(0.5, -1, 'man_R2score:\n%s'%mean)

plt.savefig('/hpc/crise/hao.c/model_result/compare_normalisation.png'%(atlas, y_file, weight, lateral))
plt.show()
"""


fig, ax = plt.subplots(1,2, figsize=(9, 4))
subdf = All_data[All_data.hemi=='lh']

fig = plt.figure(figsize=(10,9))
i = 0

for hemi in ['lh', 'rh']:
    i = i+1
    ax = fig.add_subplot(2, 1, i)
    subdf = All_data[All_data.hemi == hemi]
    bp = subdf.boxplot(column='r2', by='norma', ax=ax)

    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel('R2 score')
    ax.set_title(hemi)
    means = {id+1: round(x, 3) for id, x in enumerate(mean[hemi])}
    ax.scatter(means.keys(), means.values())
    for k, v in means.items():
        ax.text(k+0.3, v, v)


fig.texts = [] #flush the old super titles
plt.savefig('/hpc/crise/hao.c/model_result/compare_parameters/compare_normalisation/%s_%s_%s_%s.png'%(atlas, y_file, weight, lateral))
plt.show()

t_test = []
for hemisphere in ['lh', 'rh']:
    y = -0.5
    for a_index, a in enumerate(options):
        for b_index in range(a_index+1, len(options)):
            b = options[b_index]
            print a + ' vs ' + b
            a_score = All_data['r2'][(All_data.norma == a) & (All_data.hemi == hemisphere)]
            b_score = All_data['r2'][(All_data.norma == b) & (All_data.hemi == hemisphere)]

            t, p = stats.ttest_rel(a_score, b_score)
            string = cs.get_difference_ttest(a,b, t, float(p))

            t_test.append({'hemi':hemisphere, 'compare':string, 'p':p})

            if hemisphere == 'lh':
                i = 1
            else:
                i = 0

            width = b_index-a_index
            #axes[i,0].annotate( 'p: %.3E\n%s' %(p,string), xy=((b_index-a_index)/2,y), xytext=((b_index-a_index)/2,y-0.1), arrowprops=dict(arrowstyle='-[, widthB=%s, lengthB=1'%str(width), lw=1.0))
            y -=0.1

print  t_test
"""
for i in [0,1]:
    axes[i, 0].set_ylim(-1.5,)
    x = 0.5,
    #axes[i,0].annotate( t_test, xy=(x+,-0.5), xytext=(0.5, -0.6), arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1', lw=1.0))

# axes[0,0].text(5.5, -1.5, 'mean_R2score of rh:\n%s'%mean.rh)
# axes[1,0].text(5.5, -1.5, 'mean_R2score of lh:\n%s'%mean.lh)
"""
