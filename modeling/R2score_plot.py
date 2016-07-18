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
import xlrd


# read excel file and save as joblib
def save_to_jl(xlfile, sheetname, jl_file):

    book = xlrd.open_workbook(xlfile)
    sheets = book.sheets()

    for data in sheets:
        if sheetname == data.name:
            print sheetname
            subjects = data.col_values(1, 1)
            option = data.row_values(0,2)
            dict_R2 = {}
            for col, norma in enumerate(option):
                dict_R2[norma] = pd.Series(data.col_values(col+2, 1), index=subjects)

            df = pd.DataFrame(dict_R2)
            print df.shape
            print "save in jl: " +jl_file
            joblib.dump(df, jl_file)

    return df


parcel_altas = 'destrieux'
weight = 'none'
hemisphere = 'rh'
lateral = 'bilateral'
model = 'connmat'
y_file = ['rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']
# y = 'rspmT_0004'
tracto_name = '{}_STS+STG_{}_2'.format(hemisphere.upper(), parcel_altas)


# ======================================== plot score ======================================================
dict_score = []
for lateral in ['ipsi', 'bilateral']:
    for weight in ['none', 'distance']:
        for hemisphere in ['lh', 'rh']:
            for y in y_file:

                output_predict_score_excel_file = '/hpc/crise/hao.c/model_result/tracto_volume/' \
                  '%s_LOSO_R2score_compare_normalization_%s_%s_weighted_%s.xls' %(tracto_name, lateral, model, weight)

                R2_score_result_joblib = '/hpc/crise/hao.c/model_result/tracto_volume/r2score_{}_{}_{}_{}_{}_' \
                                         'weighted{}.jl'.format(hemisphere, parcel_altas, lateral, model, y, weight)

                plot_img = '/hpc/crise/hao.c/model_result/tracto_volume/r2score_{}_{}_{}_{}_{}_weighted{}.png'\
                            .format(hemisphere, parcel_altas, lateral, model, y, weight)

                if op.isfile(R2_score_result_joblib):
                    ALL_result_dataframe = joblib.load(R2_score_result_joblib)
                else:
                    ALL_result_dataframe = save_to_jl(output_predict_score_excel_file, y, R2_score_result_joblib)

                # remove AHS22 for right hemisphere:
                if hemisphere == 'rh':
                    ALL_result_dataframe = ALL_result_dataframe[1:]

                mean_score = ALL_result_dataframe.mean(axis=0)

                if not op.isfile(plot_img):
                    """
                    ALL_result_dataframe.plot()
                    plt.title('r2 %s_%s_%s_%s_weighted%s_%s' %(hemisphere, parcel_altas, model, lateral,  weight, y))
                    plt.ylabel('r2')
                    plt.xlabel('subject')
                    position = round(np.amin(ALL_result_dataframe.min(axis=0)))
                    plt.text(50, position, 'mean of R2 score: \n ' + str(mean_score))
                    plt.show()
                    plt.savefig(plot_img)
"""
                # save the mean score to datafram:
                for norma in mean_score.index:
                    d = {'altas': parcel_altas, 'hemisphere': hemisphere,'model': model, 'lateral': lateral,
                         'weight': weight, 'y_file': y, 'norma': norma, 'mean_r2': mean_score[norma]}
                    dict_score.append(d)


df = pd.DataFrame(dict_score)
print df.shape
print df
print(df.max(axis=0))
joblib.dump(df, '/hpc/crise/hao.c/model_result/tracto_volume/mean_r2_all_model.jl', compress=3)




