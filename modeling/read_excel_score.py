
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


# read excel file and save as joblib
def save_to_jl(xlfile, sheetname, jl_file):

    book = xlrd.open_workbook(xlfile)
    sheets = book.sheets()
    df = pd.DataFrame()

    for data in sheets:
        if sheetname == data.name:
            print sheetname
            subjects = data.col_values(1, 1)
            option = data.row_values(0, 2)
            dict_R2 = {}
            for col, norma in enumerate(option):
                dict_R2[norma] = pd.Series(data.col_values(col+2, 1), index=subjects)

            df = pd.DataFrame(dict_R2)
            print df.shape
            print "save in jl: " +jl_file
            joblib.dump(df, jl_file)

    return df


# ================ main =================================
parcel_altas = 'destrieux'
model = 'connmat'

dict_p = {'hemisphere': ['lh', 'rh'],
                   'normalise': ['none', 'waytotal', 'norm', 'sum', 'partarget', 'zscore'],
                   'lateral': ['ipsi', 'bilateral'],
                   'weight': ['none', 'distance'],
                   'y_file': ['rspmT_0001', 'rspmT_0002', 'rspmT_0003', 'rspmT_0004', 'rcon_0001', 'rcon_0002', 'rcon_0003', 'rcon_0004']}

dlist_score = []
for lateral in dict_p['lateral']:

    for weight in dict_p['weight']:
        for hemisphere in dict_p['hemisphere']:
            for y in dict_p['y_file']:

                tracto_name = '{}_STS+STG_{}_2'.format(hemisphere.upper(), parcel_altas)
                output_predict_score_excel_file = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux/' \
                  '%s_LOSO_R2score_compare_normalization_%s_%s_weighted_%s.xls' %(tracto_name, lateral, model, weight)

                R2_score_result_joblib = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux/r2score_{}_{}_{}_{}_{}_' \
                                         'weighted{}.jl'.format(hemisphere, parcel_altas, lateral, model, y, weight)

                ALL_result_dataframe = save_to_jl(output_predict_score_excel_file, y, R2_score_result_joblib)

                # remove AHS22 for right hemisphere:
                if hemisphere == 'rh':
                    ALL_result_dataframe = ALL_result_dataframe[1:]

                mean_score = ALL_result_dataframe.mean(axis=0)
                print "mean score of " + R2_score_result_joblib + str(mean_score)

                # save the mean score to datafram:
                for norma in dict_p['normalise']:
                    if norma in mean_score.index:
                        this_score = mean_score[norma]
                    else:
                        this_score = np.nan
                        print "not exists yet:", [hemisphere, parcel_altas, model, y , lateral, weight, norma]
                    # print this_score, norma
                    d = {'altas': parcel_altas, 'hemisphere': hemisphere,'model': model, 'lateral': lateral,
                         'weight': weight, 'y_file': y, 'norma': norma, 'mean_r2': this_score}
                    dlist_score.append(d)

data = pd.DataFrame(dlist_score)
print data.shape
print(data.max(axis=0))
joblib.dump(data, '/hpc/crise/hao.c/model_result/tracto_volume_destrieux/LOSO_mean_r2_all_model.jl', compress=3)


# ==================================== plot mean score =====================================
import seaborn as sns
hemi='lh'
LOSO_mean_score_path = '/hpc/crise/hao.c/model_result/tracto_volume_destrieux/LOSO_mean_r2_all_model.jl'
data = joblib.load(LOSO_mean_score_path)

print data.shape
# select normalization
data = data.loc[data['norma']!='waytotal']
data = data.fillna(0)

for hemi in ['lh', 'rh']:
    x = data[data.hemisphere == hemi]

    g = sns.factorplot(x="y_file", y="mean_r2", hue="norma", col="weight", row='lateral',  data=x, casize=0.2, palette="muted", size=5, aspect= 1.5)
    g.savefig('/hpc/crise/hao.c/model_result/tracto_volume_destrieux/LOSO_compare_parameter_%s.png' % hemi)


