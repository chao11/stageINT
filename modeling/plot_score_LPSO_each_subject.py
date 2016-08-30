"""
plot all R2 score of all subjects appear in 

"""

import joblib
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

hemisphere = 'lh'
parcel_altas = 'destrieux'
model = 'connmat'

score_model = joblib.load('/hpc/crise/hao.c/model_result/tracto_volume/labelshuffle_leave_P_subject_out/'
                          'r2score_lh_destrieux_ipsi_connmat_rcon_0002_weightednone.jl')


#d = defaultdict(list)
d = {}
for i in score_model:
    score = i['R2_score_test']
    for k, v in score.items():
        d.setdefault(k, []).append(v)
len(d)

i = 0
for k, v in d.items():
    print k
    plt.scatter(np.ones(len(v))*i, v, marker='.')
    i += 1

plt.xticks(range(len(d)), d.keys(), rotation=70)
plt.xlabel('subjects')
plt.ylabel('R2score')
plt.title('R2 score of subjects in the training set of leave 14 subjects out (LPSO) crosse validation\nlh_destrieux_ipsi_connmat_rcon_0002_weightednone_zscore')
plt.xlim([0,70])
plt.show()






