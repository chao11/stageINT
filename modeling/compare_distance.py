import compare_statistic as cs
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

model_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux_with_AHS22'

atlas = 'destrieux'
y_file = 'rcon_0001'
norma = 'zscore'
lateral = 'ipsi'

for y_file in ['rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']:
    lh_data, t_test_lh = cs.distance(model_score_path, 'lh', lateral, atlas, norma, y_file)
    rh_data, t_test_rh = cs.distance(model_score_path, 'rh', lateral, atlas, norma, y_file)
    All_data = pd.concat([lh_data, rh_data])

    pd.set_option('precision', 4)
    mean = All_data.groupby(['hemi', 'weight'])['r2'].mean()
    print mean
    """
    ax = sns.boxplot(x="hemi", y="r2", hue="weight", data=All_data, palette="PRGn")
    ax.set_title('compare distance/no_distance model for each hemisphere\natlas:%s, y:%s, nomra:%s, lateral:%s' %(atlas, y_file, norma, lateral))

    ax.set_ylim([-1, 0.5])
    ax.text(-0.25, 0.4, 'p= %.3f\n%s' %(t_test_lh['p'], t_test_lh['result']))
    ax.text(0.75, 0.4, 'p= %.3f\n%s' %(t_test_rh['p'], t_test_rh['result']))
    ax.text(0.5, -1, 'man_R2score:\n%s' % mean)

    #plt.savefig('/hpc/crise/hao.c/model_result/compare_parameters/compare_distance_weight/compare_distance_%s_%s.png' %(y_file, norma))
    # plt.show()
    """

    ax = All_data.boxplot(by=['hemi','weight'], column="r2", sym='')
    means = {id+1: round(x, 3) for id, x in enumerate(mean)}
    plt.scatter(means.keys(), means.values())
    for k, v in means.items():
        plt.text(k+0.1, v, v)

    #plt.text(1, 0.3, 'p= %.3f\n%s' %(t_test_lh['p'], t_test_lh['result']))
    #plt.text(3, 0.3, 'p= %.3f\n%s' %(t_test_rh['p'], t_test_rh['result']))
    plt.title('left:p= %.3f, %s;    right:p= %.3f, %s' %(t_test_lh['p'], t_test_lh['result'],t_test_rh['p'], t_test_rh['result'] ))
    plt.savefig('/hpc/crise/hao.c/model_result/compare_parameters/compare_distance_weight/withoutFlyers_compare_distance_%s_%s.png' %(y_file, norma))

    plt.show()

    print "done"


