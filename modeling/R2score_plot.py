# hao.c
# 12/07/2016


"""

select subject and plot the R2score

"""

import joblib
import os
import os.path as op
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

parcel_altas = 'destrieux'
weight = 'none'
hemisphere = 'rh'
lateral = 'ipsi'
model = 'connmat'
y_file = ['rspmT_0001','rspmT_0002','rspmT_0003','rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']
y = 'rspmT_0004'

dict_score = []
for lateral in ['ipsi', 'bilateral']:
    for weight in ['none', 'disctance']:
        for hemisphere in ['lh', 'rh']:
            for y in y_file:
                R2_score_result_joblib = '/hpc/crise/hao.c/model_result/tracto_volume/r2score_{}_{}_{}_{}_{}_' \
                                         'weighted{}.jl'.format(hemisphere, parcel_altas, lateral, model, y, weight)
                if op.isfile(R2_score_result_joblib):
                    ALL_result_dataframe = joblib.load(R2_score_result_joblib)
                    # remove AHS22 for right hemisphere:
                    if hemisphere == 'rh':
                        ALL_result_dataframe = ALL_result_dataframe[1:]

                    mean_score = ALL_result_dataframe.mean(axis=0)

                    """
                    ALL_result_dataframe.plot()
                    plt.title('r2 %s_%s_%s_%s_weighted%s_%s' %(hemisphere, parcel_altas, model, lateral,  weight, y))
                    plt.ylabel('r2')
                    plt.xlabel('subject')
                    position = round(np.amin(ALL_result_dataframe.min(axis=0)))
                    plt.text(50, position, 'mean of R2 score: \n ' + str(mean_score))
                    plt.show()
                    #plt.savefig('/hpc/crise/hao.c/model_result/tracto_volume/r2score_{}_{}_{}_{}_{}_weighted{}.png'
                                    .format(hemisphere, parcel_altas, lateral, model, y, weight))
                    """
                    # save the mean score to datafram:
                    for norma in mean_score.index:
                        d = {'altas': parcel_altas, 'hemisphere': hemisphere,'model': model, 'lateral': lateral,
                             'weight': weight, 'y_file': y, 'norma': norma, 'mean_r2': mean_score[norma]}
                        dict_score.append(d)

                else:
                    print "%s not exists " % R2_score_result_joblib

df = pd.DataFrame(dict_score)
print df.shape
